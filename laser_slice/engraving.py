"""Place vector layer-number engraving safely inside cut geometry."""
from __future__ import annotations

import warnings

import shapely.affinity as affinity
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import polylabel

from laser_slice import fonts
from laser_slice.config import Config
from laser_slice.geometry_types import EngraveGlyph

_FIT_ITERATIONS = 16
_MAX_POLYLABEL_TOLERANCE_MM = 0.1
_MIN_POLYLABEL_TOLERANCE_MM = 0.0001
_NUMERIC_INSET_MM = 0.0001
_FALLBACK_CLEARANCE_FRACTION = 0.2


class EngravingFitWarning(UserWarning):
    """A cut piece could not safely hold its layer-number engraving."""


def _iter_polygons(geometry: BaseGeometry) -> list[Polygon]:
    """Return every exterior Polygon component, never its interior rings."""
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    polygons: list[Polygon] = []
    # GeometryCollection is not expected from the pipeline, but accepting it
    # here keeps this public helper defensive.
    for subgeometry in getattr(geometry, "geoms", ()):
        polygons.extend(_iter_polygons(subgeometry))
    return polygons


def _stroke_bounds(strokes: list[EngraveGlyph]) -> tuple[float, float, float, float]:
    points = [point for stroke in strokes for point in stroke.points]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _pole(polygon: Polygon) -> tuple[float, float]:
    """Find a well-inside point with sub-millimeter, size-aware precision."""
    minx, miny, maxx, maxy = polygon.bounds
    shortest_side = min(maxx - minx, maxy - miny)
    tolerance = min(
        _MAX_POLYLABEL_TOLERANCE_MM,
        max(_MIN_POLYLABEL_TOLERANCE_MM, shortest_side / 50.0),
    )
    point = polylabel(polygon, tolerance=tolerance)
    return point.x, point.y


def _safe_area(
    component: Polygon,
    config: Config,
) -> tuple[BaseGeometry, float] | None:
    """Inset a component by the requested visible-stroke clearance.

    If a piece is too narrow for the configured margin even at a single point,
    retain a proportional inset instead of putting the label on its boundary.
    """
    stroke_radius = max(0.0, config.engrave_stroke_width_mm) / 2.0
    target_clearance = (
        max(0.0, config.engrave_margin_mm) + stroke_radius + _NUMERIC_INSET_MM
    )
    safe = component.buffer(-target_clearance)
    if _iter_polygons(safe):
        return safe, target_clearance

    pole_x, pole_y = _pole(component)
    max_clearance = component.boundary.distance(Point(pole_x, pole_y))
    minimum_visible_clearance = stroke_radius + _NUMERIC_INSET_MM
    if max_clearance <= minimum_visible_clearance:
        return None

    fallback_clearance = min(
        target_clearance,
        max(
            minimum_visible_clearance,
            max_clearance * _FALLBACK_CLEARANCE_FRACTION,
        ),
    )
    safe = component.buffer(-fallback_clearance)
    if _iter_polygons(safe):
        return safe, fallback_clearance

    return None


def _label_box(
    center: tuple[float, float],
    unit_width: float,
    unit_height: float,
    text_height: float,
) -> Polygon:
    width = unit_width * text_height
    height = unit_height * text_height
    cx, cy = center
    return box(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def _fitting_height(
    region: Polygon,
    center: tuple[float, float],
    unit_width: float,
    unit_height: float,
    requested_height: float,
) -> float:
    """Find the largest requested-or-smaller label box covered by region."""
    if region.covers(_label_box(center, unit_width, unit_height, requested_height)):
        return requested_height

    low = 0.0
    high = requested_height
    for _ in range(_FIT_ITERATIONS):
        candidate = (low + high) / 2.0
        if region.covers(_label_box(center, unit_width, unit_height, candidate)):
            low = candidate
        else:
            high = candidate

    # Stay just inside the GEOS-computed boundary rather than landing on it
    # after rounding coordinates into the SVG.
    return low * (1.0 - 0.000001)


def _safe_center(
    region: Polygon,
    unit_width: float,
    unit_height: float,
) -> tuple[float, float]:
    """Find a label-aware interior center by normalizing for text aspect ratio."""
    normalized = affinity.scale(
        region,
        xfact=1.0 / unit_width,
        yfact=1.0 / unit_height,
        origin=(0.0, 0.0),
    )
    normalized_x, normalized_y = _pole(normalized)
    return normalized_x * unit_width, normalized_y * unit_height


def _strokes_at_center(
    text: str,
    text_height: float,
    center: tuple[float, float],
    unit_bounds: tuple[float, float, float, float],
) -> list[EngraveGlyph]:
    unit_minx, unit_miny, unit_maxx, unit_maxy = unit_bounds
    unit_centerx = (unit_minx + unit_maxx) / 2.0
    unit_centery = (unit_miny + unit_maxy) / 2.0
    return fonts.text_to_strokes(
        text,
        height_mm=text_height,
        origin=(
            center[0] - unit_centerx * text_height,
            center[1] - unit_centery * text_height,
        ),
    )


def _fit_text_to_component(
    component: Polygon,
    text: str,
    requested_height: float,
    unit_bounds: tuple[float, float, float, float],
    config: Config,
) -> list[EngraveGlyph]:
    unit_minx, unit_miny, unit_maxx, unit_maxy = unit_bounds
    unit_width = unit_maxx - unit_minx
    unit_height = unit_maxy - unit_miny
    safe_result = _safe_area(component, config)
    if safe_result is None:
        return []
    safe, clearance = safe_result

    # Preserve the familiar bottom-left placement whenever the complete label
    # box really does fit there with clearance from every exterior/hole edge.
    component_minx, component_miny, _, _ = component.bounds
    lower_left = (
        component_minx + clearance + unit_width * requested_height / 2.0,
        component_miny + clearance + unit_height * requested_height / 2.0,
    )
    if safe.covers(_label_box(lower_left, unit_width, unit_height, requested_height)):
        return _strokes_at_center(text, requested_height, lower_left, unit_bounds)

    regions = _iter_polygons(safe)
    regions.sort(
        key=lambda region: min(
            requested_height,
            (region.bounds[2] - region.bounds[0]) / unit_width,
            (region.bounds[3] - region.bounds[1]) / unit_height,
        ),
        reverse=True,
    )

    best_height = 0.0
    best_center: tuple[float, float] | None = None
    for region in regions:
        upper_bound = min(
            requested_height,
            (region.bounds[2] - region.bounds[0]) / unit_width,
            (region.bounds[3] - region.bounds[1]) / unit_height,
        )
        if upper_bound <= best_height:
            continue
        center = _safe_center(region, unit_width, unit_height)
        height = _fitting_height(
            region,
            center,
            unit_width,
            unit_height,
            requested_height,
        )
        if height > best_height:
            best_height = height
            best_center = center

    if best_center is None or best_height <= 0.0:
        return []
    return _strokes_at_center(text, best_height, best_center, unit_bounds)


def layer_number_strokes(
    geometry: BaseGeometry,
    layer_index: int,
    config: Config,
) -> list[EngraveGlyph]:
    """Create one safely fitted layer number per disconnected cut piece.

    Labels retain ``engrave_text_height_mm`` where possible, move to a safe
    interior pocket for concave/holed shapes, and shrink on smaller pieces.
    Interior rings are holes and never receive their own label.
    """
    requested_height = max(0.0, config.engrave_text_height_mm)
    if requested_height == 0.0:
        return []

    text = str(layer_index)
    unit_strokes = fonts.text_to_strokes(text, height_mm=1.0)
    if not unit_strokes:
        return []
    unit_bounds = _stroke_bounds(unit_strokes)

    strokes: list[EngraveGlyph] = []
    for component_number, component in enumerate(_iter_polygons(geometry), start=1):
        component_strokes = _fit_text_to_component(
            component,
            text,
            requested_height,
            unit_bounds,
            config,
        )
        if not component_strokes:
            warnings.warn(
                f"Layer {layer_index} cut piece {component_number} is too narrow "
                "to contain the engraving stroke; its number was skipped.",
                EngravingFitWarning,
                stacklevel=2,
            )
        strokes.extend(component_strokes)
    return strokes
