"""Shared data types passed between pipeline stages.

Pipeline: mesh_io -> slicer -> polygon_ops (+ registration) -> nesting -> svg_export
                                                                     ^
                                                              visualizer reads
                                                              any stage's output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry.base import BaseGeometry


@dataclass
class LayerSlice:
    """One horizontal slab of the model, before kerf/registration processing."""
    index: int
    z_min: float
    z_max: float
    # Shapely Polygon or MultiPolygon (holes already resolved via even-odd containment).
    # None if this slab does not intersect the mesh at all.
    geometry: BaseGeometry | None

    @property
    def z_mid(self) -> float:
        return (self.z_min + self.z_max) / 2.0


@dataclass
class EngraveGlyph:
    """A single stroke (open polyline) of vector engraving, in the part's local
    coordinate frame (same frame as Part.cut_geometry, pre-placement)."""
    points: list[tuple[float, float]]


@dataclass
class Part:
    """A single layer's final cuttable geometry, ready to be nested."""
    layer_index: int
    # Final outer silhouette with holes (object holes + registration holes),
    # already kerf-compensated. Polygon or MultiPolygon.
    cut_geometry: BaseGeometry
    # Vector engraving strokes (e.g. one layer number per disconnected piece),
    # in the same local frame as cut_geometry (i.e. before any nesting
    # translation/rotation is applied).
    engrave_strokes: list[EngraveGlyph] = field(default_factory=list)
    # (x, y) model-space points (same frame as cut_geometry) where registration
    # holes were actually cut on this part (subset of requested registration points).
    registration_holes_cut: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Placement:
    """Where a Part ends up after nesting.

    Coordinate convention (CAD-style, NOT SVG): origin is the sheet's
    bottom-left corner; X increases rightward, Y increases upward.

    To materialize a Placement (this exact recipe is the shared contract
    between nesting.py, which computes x_offset_mm/y_offset_mm/rotation_deg,
    and svg_export.py, which renders the result -- both must follow it
    exactly so they agree without needing to inspect each other's code):

        geom = shapely.affinity.rotate(part.cut_geometry, rotation_deg, origin="center")
        # origin="center" = the geometry's bounding-box center (shapely default).
        minx, miny, _, _ = geom.bounds
        geom = shapely.affinity.translate(geom, xoff=x_offset_mm - minx, yoff=y_offset_mm - miny)
        # Apply the identical rotate(origin="center") + translate to every
        # point of every EngraveGlyph in part.engrave_strokes too, so the
        # engraving stays rigidly attached to the part.

    After this, geom's bounding box min corner is exactly
    (x_offset_mm, y_offset_mm) and its size is the rotated bounding-box
    size nesting.py used when it decided the placement.

    SVG note: SVG's Y axis increases downward, the opposite of this
    convention, so svg_export.py must flip every emitted Y coordinate via
    svg_y = sheet_height_mm - model_y.
    """
    part: Part
    sheet_index: int
    x_offset_mm: float
    y_offset_mm: float
    rotation_deg: float  # 0 or 90 for the bounding-box nester


@dataclass
class SheetLayout:
    sheet_index: int
    width_mm: float
    height_mm: float
    placements: list[Placement] = field(default_factory=list)
