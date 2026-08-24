"""Registration-hole placement for aligning the physical layer stack.

Registration points are computed once for the whole model (in model-space
XY, shared across all layers), then each layer independently decides which
of those points it has enough surrounding material to actually cut a hole
at.
"""
from __future__ import annotations

from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

from laser_slice.config import Config
from laser_slice.geometry_types import LayerSlice


def auto_registration_points(
    layers: list[LayerSlice], config: Config
) -> list[tuple[float, float]]:
    """Compute default registration dowel positions from the model's "core".

    If ``config.registration_points`` is explicitly set, it is returned
    as-is. Otherwise points are derived from the single layer with the
    LARGEST cross-sectional area (the "core", e.g. a torso/trunk/base),
    NOT the union of bounds across every layer. Using the union bbox of
    all layers is a trap for exactly the branching/limbed shapes this
    tool targets: a thin arm or leg sticking far out from a central body
    drags the bounding box (and therefore the naive split points) away
    from the body entirely, so the "registration holes" feature silently
    cuts zero holes on every layer -- the core is present in most layers,
    so anchoring to it gives points that are actually likely to have
    material (and required clearance) across the whole stack.

    Within the core, the required clearance (hole radius + margin) is
    eroded away first (buffer(-clearance)) to get a "safe zone". That
    zone is then split into two halves across its longer bounding-box
    dimension, and one point is taken from each half via
    half.representative_point() -- a method that is *always* guaranteed
    to return a point actually inside its polygon (or multipolygon),
    unlike picking an arbitrary bbox fraction and hoping it lands inside
    a possibly-concave shape. Splitting into independent halves (rather
    than taking one representative_point() of the whole safe zone twice)
    ensures the two dowel points are genuinely apart from each other --
    two coincident/near-coincident points would let the stack rotate
    around them, defeating the purpose of registration holes. If one
    half turns out to have no safe area at all (a very lopsided core),
    both points fall back to the whole zone's representative_point(),
    which at least still yields a hole that actually cuts, even though
    it can't lock rotation on its own.
    """
    if config.registration_points is not None:
        return config.registration_points

    candidates = [
        layer.geometry
        for layer in layers
        if layer.geometry is not None and not layer.geometry.is_empty
    ]
    if not candidates:
        return []

    core = max(candidates, key=lambda g: g.area)

    radius = config.registration_hole_diameter_mm / 2.0
    clearance = radius + config.registration_hole_margin_mm
    safe = core.buffer(-clearance)
    if safe.is_empty:
        return []

    minx, miny, maxx, maxy = safe.bounds
    width = maxx - minx
    height = maxy - miny
    pad = max(width, height, 1.0)

    if width >= height:
        mid = (minx + maxx) / 2.0
        half_a = safe.intersection(box(minx - pad, miny - pad, mid, maxy + pad))
        half_b = safe.intersection(box(mid, miny - pad, maxx + pad, maxy + pad))
    else:
        mid = (miny + maxy) / 2.0
        half_a = safe.intersection(box(minx - pad, miny - pad, maxx + pad, mid))
        half_b = safe.intersection(box(minx - pad, mid, maxx + pad, maxy + pad))

    if half_a.is_empty or half_b.is_empty:
        fallback = safe.representative_point()
        return [(fallback.x, fallback.y), (fallback.x, fallback.y)]

    p_a = half_a.representative_point()
    p_b = half_b.representative_point()
    return [(p_a.x, p_a.y), (p_b.x, p_b.y)]


def apply_registration_holes(
    geometry: BaseGeometry | None,
    points: list[tuple[float, float]],
    config: Config,
) -> tuple[BaseGeometry | None, list[tuple[float, float]]]:
    """Cut registration holes into ``geometry`` at qualifying points.

    Points are processed in order against the current (possibly
    already-updated-by-a-prior-point) geometry. A point qualifies if it
    lies within the geometry and is at least ``radius +
    registration_hole_margin_mm`` away from the geometry's boundary (outer
    edge or any existing hole edge). Holes are cut at nominal diameter;
    kerf compensation happens later via a single shared apply_kerf() call.

    Returns (updated_geometry, points_actually_cut).
    """
    if geometry is None or not config.registration_holes_enabled or not points:
        return geometry, []

    radius = config.registration_hole_diameter_mm / 2.0
    required_clearance = radius + config.registration_hole_margin_mm

    cut_points: list[tuple[float, float]] = []
    current = geometry

    for x, y in points:
        p = Point(x, y)
        if current.contains(p) and current.boundary.distance(p) >= required_clearance:
            current = current.difference(p.buffer(radius))
            cut_points.append((x, y))

    return current, cut_points
