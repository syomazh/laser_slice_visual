from __future__ import annotations

import math

from laser_slice.fonts import text_to_strokes


def _bbox(glyphs):
    xs = [x for g in glyphs for x, y in g.points]
    ys = [y for g in glyphs for x, y in g.points]
    return min(xs), min(ys), max(xs), max(ys)


def test_text_to_strokes_nonempty():
    glyphs = text_to_strokes("0123456789", height_mm=5.0)
    assert len(glyphs) > 0
    for g in glyphs:
        assert len(g.points) >= 1


def test_text_to_strokes_bbox_height_close_to_requested():
    height_mm = 5.0
    glyphs = text_to_strokes("0123456789", height_mm=height_mm)
    minx, miny, maxx, maxy = _bbox(glyphs)
    height = maxy - miny
    assert math.isclose(height, height_mm, rel_tol=0.05)


def test_digits_are_not_identical_point_sets():
    height_mm = 5.0
    per_digit_points = {}
    for d in "0123456789":
        glyphs = text_to_strokes(d, height_mm=height_mm)
        pts = frozenset(pt for g in glyphs for pt in g.points)
        per_digit_points[d] = pts

    digits = list(per_digit_points.keys())
    for i in range(len(digits)):
        for j in range(i + 1, len(digits)):
            a, b = digits[i], digits[j]
            assert per_digit_points[a] != per_digit_points[b], f"{a} and {b} have identical point sets"


def test_origin_offsets_layout():
    origin = (10.0, 20.0)
    glyphs = text_to_strokes("1", height_mm=4.0, origin=origin)
    xs = [x for g in glyphs for x, y in g.points]
    ys = [y for g in glyphs for x, y in g.points]
    assert min(xs) >= origin[0] - 1e-9
    assert min(ys) >= origin[1] - 1e-9


def test_unsupported_character_renders_placeholder_without_raising():
    glyphs = text_to_strokes("A?", height_mm=5.0)
    assert len(glyphs) > 0
    for g in glyphs:
        assert len(g.points) >= 2


def test_advance_positions_characters_left_to_right():
    glyphs_single = text_to_strokes("1", height_mm=5.0)
    glyphs_double = text_to_strokes("11", height_mm=5.0)
    max_x_single = max(x for g in glyphs_single for x, y in g.points)
    max_x_double = max(x for g in glyphs_double for x, y in g.points)
    assert max_x_double > max_x_single
