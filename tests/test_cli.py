from __future__ import annotations

import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from laser_slice import fonts
from laser_slice.config import Config
from laser_slice.engraving import EngravingFitWarning, layer_number_strokes

TOLERANCE_MM = 0.00001


def _stroke_lines(strokes):
    return [LineString(stroke.points) for stroke in strokes if len(stroke.points) >= 2]


def _linework(strokes):
    return unary_union(_stroke_lines(strokes))


def _assert_linework_safe(
    engraving,
    polygon: Polygon,
    centerline_clearance_mm: float,
    stroke_radius_mm: float,
) -> None:
    assert polygon.buffer(TOLERANCE_MM).covers(engraving)
    assert polygon.boundary.distance(engraving) >= centerline_clearance_mm - TOLERANCE_MM
    assert polygon.buffer(-centerline_clearance_mm).buffer(TOLERANCE_MM).covers(engraving)
    assert polygon.buffer(TOLERANCE_MM).covers(engraving.buffer(stroke_radius_mm))


def _assert_safe(strokes, polygon: Polygon, config: Config) -> None:
    stroke_radius = config.engrave_stroke_width_mm / 2.0
    _assert_linework_safe(
        _linework(strokes),
        polygon,
        config.engrave_margin_mm + stroke_radius,
        stroke_radius,
    )


def test_layer_number_strokes_labels_each_disconnected_piece() -> None:
    components = [
        box(0, 0, 20, 20),
        box(40, 5, 65, 30),
        box(80, -4, 108, 24),
    ]
    geometry = MultiPolygon(components)
    config = Config(engrave_text_height_mm=4.0, engrave_margin_mm=3.0)

    actual = layer_number_strokes(geometry, layer_index=12, config=config)

    strokes_per_label = len(fonts.text_to_strokes("12", height_mm=1.0))
    assert len(actual) == len(components) * strokes_per_label
    lines = _stroke_lines(actual)
    for component in components:
        component_lines = [line for line in lines if component.covers(line)]
        assert len(component_lines) == strokes_per_label
        _assert_linework_safe(
            unary_union(component_lines),
            component,
            config.engrave_margin_mm + config.engrave_stroke_width_mm / 2.0,
            config.engrave_stroke_width_mm / 2.0,
        )
        component_engraving = unary_union(component_lines)
        assert component_engraving.bounds[3] - component_engraving.bounds[1] == pytest.approx(4.0)
    assert all(sum(component.covers(line) for component in components) == 1 for line in lines)


def test_layer_number_strokes_moves_label_inside_concave_piece() -> None:
    geometry = Polygon([(0, 12), (12, 12), (12, 0), (32, 0), (32, 32), (0, 32)])
    config = Config(engrave_text_height_mm=4.0, engrave_margin_mm=3.0)
    old_placement = fonts.text_to_strokes("0", height_mm=4.0, origin=(3.0, 3.0))
    assert not geometry.covers(_linework(old_placement))

    actual = layer_number_strokes(geometry, layer_index=0, config=config)

    assert len(actual) == 1
    _assert_safe(actual, geometry, config)


def test_layer_number_strokes_avoids_hole_and_does_not_label_it() -> None:
    outer = box(0, 0, 40, 30)
    hole = box(2, 2, 18, 14)
    geometry = Polygon(outer.exterior.coords, [hole.exterior.coords])
    config = Config(engrave_text_height_mm=4.0, engrave_margin_mm=3.0)
    old_placement = fonts.text_to_strokes("0", height_mm=4.0, origin=(3.0, 3.0))
    assert not geometry.covers(_linework(old_placement))

    actual = layer_number_strokes(geometry, layer_index=0, config=config)

    assert len(actual) == 1
    _assert_safe(actual, geometry, config)


def test_layer_number_strokes_scales_down_for_small_piece() -> None:
    geometry = box(0, 0, 8, 6)
    config = Config(engrave_text_height_mm=8.0, engrave_margin_mm=1.0)

    actual = layer_number_strokes(geometry, layer_index=0, config=config)

    assert len(actual) == 1
    _assert_safe(actual, geometry, config)
    engraving_height = _linework(actual).bounds[3] - _linework(actual).bounds[1]
    assert 3.84 < engraving_height < 3.86


def test_layer_number_strokes_reduces_clearance_when_piece_is_too_small() -> None:
    geometry = box(0, 0, 2, 1)
    config = Config(engrave_text_height_mm=4.0, engrave_margin_mm=1.0)
    target_clearance = config.engrave_margin_mm + config.engrave_stroke_width_mm / 2.0
    assert geometry.buffer(-target_clearance).is_empty

    actual = layer_number_strokes(geometry, layer_index=0, config=config)

    assert len(actual) == 1
    engraving = _linework(actual)
    fallback_clearance = geometry.boundary.distance(engraving)
    assert fallback_clearance == pytest.approx(0.1, abs=0.0002)
    _assert_linework_safe(
        engraving,
        geometry,
        fallback_clearance,
        config.engrave_stroke_width_mm / 2.0,
    )


def test_layer_number_strokes_warns_and_skips_impossibly_thin_piece() -> None:
    geometry = box(0, 0, 2, 0.1)
    config = Config(engrave_text_height_mm=4.0, engrave_margin_mm=1.0)

    with pytest.warns(EngravingFitWarning, match="too narrow"):
        actual = layer_number_strokes(geometry, layer_index=0, config=config)

    assert actual == []
