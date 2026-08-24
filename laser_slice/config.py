"""Configuration for a laser-slice job.

All internal geometry is in millimeters, regardless of the input mesh's
native units -- ``model_height_mm`` / ``scale_factor`` control how the
source mesh is mapped into mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- material / slicing ---
    material_thickness_mm: float = 3.0
    """Thickness of one physical sheet of material (default: 3mm basswood)."""

    slice_axis: str = "z"
    """Axis to stack layers along: 'x', 'y', or 'z'. This is the model's up-axis."""

    model_height_mm: float | None = None
    """If set, uniformly scale the mesh so its extent along slice_axis equals this value."""

    scale_factor: float | None = None
    """Alternative to model_height_mm: multiply raw mesh coordinates by this factor.
    Exactly one of model_height_mm / scale_factor should be set; if both are None,
    the mesh's native coordinates are assumed to already be in millimeters."""

    kerf_mm: float = 0.0
    """Total width of material the laser beam removes. Cut paths are offset by
    kerf_mm / 2 so the finished (post-cut) part matches the nominal design size."""

    # --- sheet / nesting ---
    sheet_width_mm: float = 300.0
    sheet_height_mm: float = 600.0
    sheet_margin_mm: float = 5.0
    """Keep-out margin from the sheet edges."""
    part_spacing_mm: float = 3.0
    """Minimum gap enforced between nested parts (and between parts and margin)."""
    allow_rotation: bool = True
    """Allow parts to be rotated 90 degrees during nesting if it improves packing."""

    # --- registration holes (for aligning the physical stack with dowels) ---
    registration_holes_enabled: bool = True
    registration_hole_diameter_mm: float = 3.0
    registration_hole_margin_mm: float = 4.0
    """Minimum required distance from a candidate hole center to any polygon
    boundary (outer edge or existing hole edge) for the hole to be cut on a
    given layer. Layers where a registration point doesn't have enough
    surrounding material simply skip that hole."""
    registration_points: list[tuple[float, float]] | None = None
    """Explicit (x, y) points in model space (post-scaling, mm) to use as dowel
    positions. If None, two points are auto-computed from the overall model
    XY footprint (see registration.auto_registration_points)."""

    # --- engraving ---
    engrave_enabled: bool = True
    engrave_text_height_mm: float = 4.0
    """Maximum layer-number height; labels shrink to fit smaller cut pieces."""
    engrave_margin_mm: float = 3.0
    """Target clearance from a layer number to exterior and hole boundaries.
    Automatically reduced only when a cut piece is too narrow to preserve it."""
    engrave_stroke_width_mm: float = 0.15

    # --- output ---
    output_dir: str = "output"

    def slice_axis_index(self) -> int:
        return {"x": 0, "y": 1, "z": 2}[self.slice_axis.lower()]
