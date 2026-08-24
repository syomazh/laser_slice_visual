from __future__ import annotations

import pytest
from shapely.geometry import Point, Polygon

from laser_slice.polygon_ops import apply_kerf, clean_geometry


def _square(cx=0.0, cy=0.0, side=10.0):
    h = side / 2.0
    return Polygon(
        [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    )


def test_apply_kerf_zero_or_negative_returns_unchanged():
    square = _square()
    assert apply_kerf(square, 0.0) is square
    assert apply_kerf(square, -1.0) is square


def test_apply_kerf_none_returns_none():
    assert apply_kerf(None, 1.0) is None


def test_apply_kerf_grows_outer_boundary():
    square = _square()  # 10x10 centered at origin -> bounds (-5,-5,5,5)
    result = apply_kerf(square, kerf_mm=1.0)
    minx, miny, maxx, maxy = result.bounds
    assert minx == pytest.approx(-5.5, abs=1e-6)
    assert miny == pytest.approx(-5.5, abs=1e-6)
    assert maxx == pytest.approx(5.5, abs=1e-6)
    assert maxy == pytest.approx(5.5, abs=1e-6)


def test_apply_kerf_shrinks_hole():
    outer = _square()
    hole = Point(0, 0).buffer(1.0)  # 2mm-diameter circular hole at center
    with_hole = outer.difference(hole)

    result = apply_kerf(with_hole, kerf_mm=1.0)

    # A point exactly on the original hole boundary should now be covered
    # by material (i.e. inside the solid, not inside any interior ring).
    boundary_point = Point(1.0, 0.0)
    assert result.contains(boundary_point)
    # Sanity: none of the interior rings still contain this point as a hole.
    if result.geom_type == "Polygon":
        interiors = result.interiors
    else:
        interiors = [ring for poly in result.geoms for ring in poly.interiors]
    for ring in interiors:
        assert not Polygon(ring).contains(boundary_point)


def test_clean_geometry_none_and_empty():
    assert clean_geometry(None) is None
    assert clean_geometry(Polygon()) is None


def test_clean_geometry_drops_tiny_sliver():
    # A well-formed 10x10 square plus a tiny sliver triangle far away,
    # combined into a MultiPolygon-producing self-touching shape via union.
    big = _square(side=10.0)
    tiny = Polygon([(100, 100), (100.05, 100), (100, 100.05)])  # area ~0.00125 mm^2
    assert tiny.area < 0.01

    combined = big.union(tiny)
    cleaned = clean_geometry(combined)

    assert cleaned is not None
    # Only the big square should remain.
    assert cleaned.area == pytest.approx(big.area, rel=1e-6)


def test_clean_geometry_returns_none_when_everything_is_tiny():
    tiny = Polygon([(0, 0), (0.05, 0), (0, 0.05)])
    assert tiny.area < 0.01
    assert clean_geometry(tiny) is None
