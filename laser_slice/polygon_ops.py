"""Geometry cleanup and kerf-compensation helpers.

These operate on shapely Polygon/MultiPolygon geometry produced by the
slicer stage, before nesting.
"""
from __future__ import annotations

from shapely.geometry.base import BaseGeometry
from shapely.geometry import MultiPolygon, Polygon

_MIN_AREA_MM2 = 0.01


def clean_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
    """Fix minor self-intersections/invalidity and drop tiny sliver sub-polygons.

    Returns None if nothing of significant area remains.
    """
    if geometry is None or geometry.is_empty:
        return None

    fixed = geometry.buffer(0)

    if fixed.is_empty:
        return None

    if isinstance(fixed, Polygon):
        polys = [fixed]
    elif isinstance(fixed, MultiPolygon):
        polys = list(fixed.geoms)
    else:
        # buffer(0) on a valid polygonal geometry should always yield a
        # Polygon or MultiPolygon; guard defensively for anything else.
        polys = [g for g in getattr(fixed, "geoms", [fixed]) if isinstance(g, Polygon)]

    kept = [p for p in polys if not p.is_empty and p.area >= _MIN_AREA_MM2]

    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return MultiPolygon(kept)


def apply_kerf(geometry: BaseGeometry | None, kerf_mm: float) -> BaseGeometry | None:
    """Dilate geometry by kerf_mm / 2 to compensate for laser kerf.

    A laser kerf removes a strip of width kerf_mm centered on the drawn cut
    path. Drawing the cut exactly on the nominal boundary would shrink the
    finished part by kerf_mm/2 at every edge -- outer boundaries shrink
    inward AND holes grow bigger. To compensate, dilate the whole
    polygon-with-holes (exterior + interior rings together, in one buffer()
    call) by +kerf_mm/2: this pushes outer boundaries outward by kerf_mm/2
    and shrinks every hole boundary inward by kerf_mm/2.
    """
    if kerf_mm <= 0 or geometry is None:
        return geometry
    return geometry.buffer(kerf_mm / 2.0, join_style="round")
