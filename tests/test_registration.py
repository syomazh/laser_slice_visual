from __future__ import annotations

import math

import pytest
from shapely.geometry import Point, Polygon, box

from laser_slice.config import Config
from laser_slice.geometry_types import LayerSlice
from laser_slice.registration import apply_registration_holes, auto_registration_points


def test_auto_registration_points_uses_explicit_config():
    explicit = [(1.0, 2.0), (3.0, 4.0)]
    config = Config(registration_points=explicit)
    layers = [LayerSlice(index=0, z_min=0, z_max=1, geometry=box(0, 0, 100, 20))]
    assert auto_registration_points(layers, config) == explicit


def test_auto_registration_points_on_100x20_rectangle():
    config = Config()  # default diameter 3.0mm, margin 4.0mm -> clearance 5.5mm
    core = box(0, 0, 100, 20)
    layers = [
        LayerSlice(index=0, z_min=0, z_max=1, geometry=core),
        LayerSlice(index=1, z_min=1, z_max=2, geometry=None),
    ]
    points = auto_registration_points(layers, config)
    assert len(points) == 2

    clearance = config.registration_hole_diameter_mm / 2.0 + config.registration_hole_margin_mm
    xs = sorted(p[0] for p in points)
    # One point in the left half, one in the right half (not coincident --
    # two dowel points that landed on top of each other couldn't stop the
    # stack from rotating around them), both with the required clearance.
    assert xs[0] < 50.0 < xs[1]
    for x, y in points:
        assert core.contains(Point(x, y))
        assert core.boundary.distance(Point(x, y)) >= clearance
        assert y == pytest.approx(10.0, abs=2.0)


def test_auto_registration_points_ignores_outlier_limb_layers():
    # Regression test for a real bug found in examples/sample.obj: a big
    # central "core" (e.g. a torso) present in most layers, plus a thin
    # "limb" layer sticking out far to one side (present in only a few
    # layers). The union-of-all-bounds heuristic this replaced would drag
    # the 25%/75% points out towards the limb, landing them outside the
    # core entirely and cutting zero registration holes on every layer.
    config = Config()
    core = box(-20, -20, 20, 20)  # 40x40 "torso", area 1600
    limb = box(40, -5, 70, 5)  # thin "arm" far to the right, area 300
    layers = [
        LayerSlice(index=0, z_min=0, z_max=1, geometry=core),
        LayerSlice(index=1, z_min=1, z_max=2, geometry=limb),
    ]
    points = auto_registration_points(layers, config)
    assert len(points) == 2
    for x, y in points:
        # Both points must land well inside the core, nowhere near the limb.
        assert core.contains(Point(x, y))
        assert core.boundary.distance(Point(x, y)) >= (
            config.registration_hole_diameter_mm / 2.0 + config.registration_hole_margin_mm
        )


def test_apply_registration_holes_none_or_disabled_or_no_points():
    config = Config()
    square = box(0, 0, 100, 100)
    assert apply_registration_holes(None, [(50, 50)], config) == (None, [])

    disabled_config = Config(registration_holes_enabled=False)
    geom, cut = apply_registration_holes(square, [(50, 50)], disabled_config)
    assert geom is square
    assert cut == []

    geom, cut = apply_registration_holes(square, [], config)
    assert geom is square
    assert cut == []


def test_apply_registration_holes_cuts_both_default_holes():
    config = Config()  # default diameter 3.0mm, margin 4.0mm
    square = box(0, 0, 100, 100)
    points = [(25.0, 50.0), (75.0, 50.0)]

    result_geom, cut_points = apply_registration_holes(square, points, config)

    assert cut_points == points
    radius = config.registration_hole_diameter_mm / 2.0
    expected_area = square.area - 2 * math.pi * radius**2
    assert result_geom.area == pytest.approx(expected_area, rel=1e-3)

    # Each requested point should now be inside a hole (not covered by
    # material) since we cut at nominal diameter around it.
    for x, y in points:
        assert not result_geom.contains(Point(x, y))


def test_apply_registration_holes_rejects_point_too_close_to_edge():
    config = Config()  # radius 1.5, margin 4.0 -> required clearance 5.5mm
    square = box(0, 0, 100, 100)

    # This point is only 3mm from the left edge, less than the 5.5mm
    # required clearance, so it must be skipped.
    too_close = (3.0, 50.0)
    far_enough = (50.0, 50.0)
    points = [too_close, far_enough]

    result_geom, cut_points = apply_registration_holes(square, points, config)

    assert too_close not in cut_points
    assert far_enough in cut_points
    assert len(cut_points) == 1

    # Geometry near the rejected point is unchanged (still fully solid).
    assert result_geom.contains(Point(*too_close))
