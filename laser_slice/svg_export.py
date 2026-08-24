"""Render nested SheetLayouts to LightBurn-ready SVG files.

Coordinate handling here must match the exact contract documented on
``laser_slice.geometry_types.Placement``: rotate each part's cut geometry
(and its engrave strokes, by the identical rigid transform) about the
cut geometry's own bounding-box center, then translate so the rotated
bounding box's min corner lands at (x_offset_mm, y_offset_mm). Finally,
because SVG's Y axis increases downward (the opposite of the Placement
convention), every emitted Y coordinate is flipped via
``svg_y = sheet.height_mm - model_y``.
"""
from __future__ import annotations

import os

import shapely.affinity as affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
import svgwrite

from laser_slice.config import Config
from laser_slice.geometry_types import Placement, SheetLayout

Coord = tuple[float, float]


def _iter_polygons(geom) -> list[Polygon]:
    """Yield the individual Polygon objects making up geom (Polygon or MultiPolygon)."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    # Fall back for any other multi-geometry container that happens to hold polygons.
    if hasattr(geom, "geoms"):
        polys: list[Polygon] = []
        for sub in geom.geoms:
            polys.extend(_iter_polygons(sub))
        return polys
    return []


def _materialize_placement(placement: Placement):
    """Apply the Placement contract's rotate(origin=center)+translate recipe.

    Returns (cut_geometry_in_sheet_space, list_of_glyph_point_lists_in_sheet_space),
    both still in the CAD-style (Y-up) convention -- the SVG Y-flip is applied
    separately at emission time.
    """
    part = placement.part
    cut_geom = part.cut_geometry

    minx0, miny0, maxx0, maxy0 = cut_geom.bounds
    center = ((minx0 + maxx0) / 2.0, (miny0 + maxy0) / 2.0)

    rotated_cut = affinity.rotate(cut_geom, placement.rotation_deg, origin=center)
    minx, miny, _, _ = rotated_cut.bounds
    xoff = placement.x_offset_mm - minx
    yoff = placement.y_offset_mm - miny
    final_cut = affinity.translate(rotated_cut, xoff=xoff, yoff=yoff)

    final_strokes: list[list[Coord]] = []
    for glyph in part.engrave_strokes:
        pts = glyph.points
        if not pts:
            final_strokes.append([])
            continue
        # Apply the identical rigid transform (same rotation origin/angle and
        # same translation offset used for cut_geom) to every point, so the
        # engraving stays rigidly attached to the part regardless of the
        # glyph's own bounding box.
        if len(pts) == 1:
            shape = Point(pts[0])
        else:
            shape = LineString(pts)
        shape = affinity.rotate(shape, placement.rotation_deg, origin=center)
        shape = affinity.translate(shape, xoff=xoff, yoff=yoff)
        final_strokes.append(list(shape.coords))

    return final_cut, final_strokes


def _ring_subpath(coords, sheet_height_mm: float) -> str:
    """Build one 'M...L...Z' subpath from a ring's coordinates, flipping Y for SVG."""
    pts = [(x, sheet_height_mm - y) for x, y in coords]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not pts:
        return ""
    parts = [f"M {pts[0][0]:.6f} {pts[0][1]:.6f}"]
    for x, y in pts[1:]:
        parts.append(f"L {x:.6f} {y:.6f}")
    parts.append("Z")
    return " ".join(parts)


def _polygon_path_d(poly: Polygon, sheet_height_mm: float) -> str:
    """Combine exterior + interior (hole) rings into one path 'd' attribute."""
    subpaths = [_ring_subpath(poly.exterior.coords, sheet_height_mm)]
    for interior in poly.interiors:
        subpaths.append(_ring_subpath(interior.coords, sheet_height_mm))
    return " ".join(s for s in subpaths if s)


def _polyline_path_d(points: list[Coord], sheet_height_mm: float) -> str:
    """Build an open 'M...L...' path (no Z) from a stroke's points, flipping Y."""
    pts = [(x, sheet_height_mm - y) for x, y in points]
    if not pts:
        return ""
    parts = [f"M {pts[0][0]:.6f} {pts[0][1]:.6f}"]
    for x, y in pts[1:]:
        parts.append(f"L {x:.6f} {y:.6f}")
    return " ".join(parts)


def export_sheets(sheets: list[SheetLayout], config: Config, out_dir: str) -> list[str]:
    """Write one SVG file per SheetLayout into out_dir, returning the file paths written."""
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []

    for index, sheet in enumerate(sheets):
        filename = f"sheet_{index + 1:02d}.svg"
        filepath = os.path.join(out_dir, filename)

        dwg = svgwrite.Drawing(
            filepath,
            size=(f"{sheet.width_mm}mm", f"{sheet.height_mm}mm"),
            viewBox=f"0 0 {sheet.width_mm} {sheet.height_mm}",
        )
        for placement in sheet.placements:
            cut_geom, strokes = _materialize_placement(placement)
            layer_id = f"layer-{placement.part.layer_index}"
            layer_group = dwg.g(id=layer_id)
            cut_group = dwg.g(
                id=f"{layer_id}-cut",
                class_="cut",
                stroke="#FF0000",
                fill="none",
                stroke_width=0.1,
            )
            engrave_group = dwg.g(
                id=f"{layer_id}-engrave",
                class_="engrave",
                stroke="#0000FF",
                fill="none",
                stroke_width=config.engrave_stroke_width_mm,
                stroke_linecap="round",
                stroke_linejoin="round",
            )

            for poly in _iter_polygons(cut_geom):
                d = _polygon_path_d(poly, sheet.height_mm)
                if d:
                    cut_group.add(dwg.path(d=d))

            for stroke_points in strokes:
                d = _polyline_path_d(stroke_points, sheet.height_mm)
                if d:
                    engrave_group.add(dwg.path(d=d))

            layer_group.add(cut_group)
            layer_group.add(engrave_group)
            dwg.add(layer_group)

        dwg.save()
        written.append(filepath)

    return written
