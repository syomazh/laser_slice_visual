"""Command-line entry point wiring the full laser_slice pipeline together.

Pipeline order (see laser_slice/geometry_types.py for the shared data
contracts each stage passes along):

    mesh_io.load_mesh
      -> slicer.slice_mesh
      -> polygon_ops.clean_geometry            (per layer)
      -> registration.auto_registration_points (once, whole job)
      -> registration.apply_registration_holes (per layer)
      -> build Part(...)
      -> fonts.text_to_strokes                 (per part, before nesting,
                                                 so the label rigidly moves
                                                 with the part)
      -> polygon_ops.apply_kerf                (per part, LAST geometric
                                                 step, after registration
                                                 holes are already cut)
      -> nesting.nest_parts
      -> svg_export.export_sheets
      -> visualizer.render_overview (always)
      -> stack_preview.build_stack_mesh + visualizer.render_stack_preview
         (+ export stack_preview.obj)                (unless --no-stack-preview)
      -> visualizer.show_interactive (unless --no-visualize)
      -> visualizer.show_stack_preview (unless --no-visualize or --no-stack-preview)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

from laser_slice import (
    fonts,
    mesh_io,
    nesting,
    polygon_ops,
    registration,
    slicer,
    stack_preview,
    svg_export,
    visualizer,
)
from laser_slice.config import Config
from laser_slice.geometry_types import Part

_SHEET_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*[xX]\s*([0-9]*\.?[0-9]+)\s*$")


def _parse_sheet(value: str) -> tuple[float, float]:
    match = _SHEET_RE.match(value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid --sheet value {value!r}; expected WIDTHxHEIGHT, e.g. 300x600"
        )
    return float(match.group(1)), float(match.group(2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laser-slice",
        description="Turn a 3D model into stacked, laser-cuttable horizontal layers (SVG, auto-nested).",
    )

    parser.add_argument("input", metavar="INPUT.obj", help="Path to the input 3D model file.")

    parser.add_argument(
        "--thickness", type=float, default=3.0, metavar="MM",
        help="Material thickness in mm (default: 3.0).",
    )
    parser.add_argument(
        "--sheet", type=_parse_sheet, default=(300.0, 600.0), metavar="WxH",
        help="Sheet size in mm as WIDTHxHEIGHT (default: 300x600).",
    )
    parser.add_argument(
        "--margin", type=float, default=5.0, metavar="MM",
        help="Keep-out margin from sheet edges, in mm (default: 5.0).",
    )
    parser.add_argument(
        "--spacing", type=float, default=3.0, metavar="MM",
        help="Minimum gap between nested parts, in mm (default: 3.0).",
    )
    parser.add_argument(
        "--kerf", type=float, default=0.0, metavar="MM",
        help="Total laser kerf width in mm (default: 0.0, no compensation).",
    )
    parser.add_argument(
        "--axis", choices=["x", "y", "z"], default="z",
        help="Model axis to slice/stack along (default: z).",
    )

    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--height-mm", type=float, default=None, metavar="H",
        help="Uniformly scale the mesh so its extent along --axis equals H mm.",
    )
    scale_group.add_argument(
        "--scale", type=float, default=None, metavar="S",
        help="Uniformly scale raw mesh coordinates by S (alternative to --height-mm).",
    )

    parser.add_argument(
        "--no-rotation", action="store_true",
        help="Disable 90-degree part rotation during nesting.",
    )

    parser.add_argument(
        "--no-registration", action="store_true",
        help="Disable registration (dowel) holes.",
    )
    parser.add_argument(
        "--reg-diameter", type=float, default=3.0, metavar="MM",
        help="Registration hole diameter in mm (default: 3.0).",
    )
    parser.add_argument(
        "--reg-margin", type=float, default=4.0, metavar="MM",
        help="Minimum required clearance from a registration hole to any boundary, in mm (default: 4.0).",
    )

    parser.add_argument(
        "--no-engrave", action="store_true",
        help="Disable engraving of layer numbers.",
    )
    parser.add_argument(
        "--engrave-height", type=float, default=4.0, metavar="MM",
        help="Engraved layer-number text height in mm (default: 4.0).",
    )
    parser.add_argument(
        "--engrave-margin", type=float, default=3.0, metavar="MM",
        help="Distance from a part's bounding box edge to the engraved layer number, in mm (default: 3.0).",
    )

    parser.add_argument(
        "--out", default="output", metavar="DIR",
        help="Output directory for SVGs and the overview image (default: output/).",
    )

    parser.add_argument(
        "--no-visualize", action="store_true",
        help="Skip the interactive matplotlib viewer at the end of the run.",
    )
    parser.add_argument(
        "--no-stack-preview", action="store_true",
        help="Skip building the physically-reconstructed stack preview (PNG + OBJ).",
    )

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    sheet_w, sheet_h = args.sheet
    return Config(
        material_thickness_mm=args.thickness,
        slice_axis=args.axis,
        model_height_mm=args.height_mm,
        scale_factor=args.scale,
        kerf_mm=args.kerf,
        sheet_width_mm=sheet_w,
        sheet_height_mm=sheet_h,
        sheet_margin_mm=args.margin,
        part_spacing_mm=args.spacing,
        allow_rotation=not args.no_rotation,
        registration_holes_enabled=not args.no_registration,
        registration_hole_diameter_mm=args.reg_diameter,
        registration_hole_margin_mm=args.reg_margin,
        engrave_enabled=not args.no_engrave,
        engrave_text_height_mm=args.engrave_height,
        engrave_margin_mm=args.engrave_margin,
        output_dir=args.out,
    )


def run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)

    mesh = mesh_io.load_mesh(args.input, config)
    layers = slicer.slice_mesh(mesh, config)

    registration_points = registration.auto_registration_points(layers, config)

    parts: list[Part] = []
    skipped_layers: list[int] = []
    layers_with_missed_holes: list[tuple[int, int, int]] = []

    for layer in layers:
        if layer.geometry is None:
            skipped_layers.append(layer.index)
            continue

        geometry = polygon_ops.clean_geometry(layer.geometry)
        if geometry is None:
            skipped_layers.append(layer.index)
            continue

        geometry, cut_points = registration.apply_registration_holes(
            geometry, registration_points, config
        )

        part = Part(
            layer_index=layer.index,
            cut_geometry=geometry,
            registration_holes_cut=cut_points,
        )

        if registration_points and len(cut_points) < len(registration_points):
            layers_with_missed_holes.append(
                (layer.index, len(registration_points), len(cut_points))
            )

        if config.engrave_enabled:
            bbox_minx, bbox_miny, _, _ = part.cut_geometry.bounds
            part.engrave_strokes = fonts.text_to_strokes(
                str(layer.index),
                height_mm=config.engrave_text_height_mm,
                origin=(
                    bbox_minx + config.engrave_margin_mm,
                    bbox_miny + config.engrave_margin_mm,
                ),
            )

        part.cut_geometry = polygon_ops.apply_kerf(part.cut_geometry, config.kerf_mm)

        parts.append(part)

    if skipped_layers:
        print(
            f"Warning: {len(skipped_layers)} layer(s) had no material and were skipped: "
            f"{skipped_layers}",
            file=sys.stderr,
        )

    sheets = nesting.nest_parts(parts, config)
    svg_paths = svg_export.export_sheets(sheets, config, config.output_dir)

    overview_path = os.path.join(config.output_dir, "overview.png")
    visualizer.render_overview(mesh, layers, sheets, config, overview_path)

    stack_preview_png = None
    stack_preview_obj = None
    if not args.no_stack_preview:
        stack_mesh = stack_preview.build_stack_mesh(parts, layers)
        stack_preview_png = os.path.join(config.output_dir, "stack_preview.png")
        visualizer.render_stack_preview(stack_mesh, stack_preview_png)
        if len(stack_mesh.faces) > 0:
            stack_preview_obj = os.path.join(config.output_dir, "stack_preview.obj")
            stack_mesh.export(stack_preview_obj)

    if not args.no_visualize:
        visualizer.show_interactive(mesh, layers, sheets, config)
        if not args.no_stack_preview:
            visualizer.show_stack_preview(stack_mesh)

    print()
    print("=== laser_slice summary ===")
    print(f"Total layers:        {len(layers)}")
    print(
        f"Layers with no material (skipped): {len(skipped_layers)} "
        f"{skipped_layers if skipped_layers else ''}"
    )
    print(f"Parts nested:        {len(parts)}")
    print(f"Sheets needed:       {len(sheets)}")
    if registration_points:
        print(f"Registration points ({len(registration_points)}): {registration_points}")
        if layers_with_missed_holes:
            print(
                "Layers with a registration hole requested but skipped "
                "(not enough surrounding material):"
            )
            for layer_index, requested, cut in layers_with_missed_holes:
                print(f"  layer {layer_index}: cut {cut}/{requested}")
        else:
            print("All registration holes were cut on every eligible layer.")
    else:
        print("Registration points: none (disabled or no geometry found).")
    print(f"Overview image:      {overview_path}")
    if stack_preview_png:
        print(f"Stack preview image: {stack_preview_png}")
    if stack_preview_obj:
        print(f"Stack preview mesh:  {stack_preview_obj}")
    print(f"SVG sheets ({len(svg_paths)}):")
    for path in svg_paths:
        print(f"  {path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        return run(args)
    except Exception as exc:
        print(f"laser-slice: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
