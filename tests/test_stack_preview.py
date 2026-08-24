from __future__ import annotations

import pytest
from shapely.geometry import Point, box

from laser_slice.geometry_types import LayerSlice, Part
from laser_slice.stack_preview import build_stack_mesh


def test_build_stack_mesh_two_simple_slabs():
    layers = [
        LayerSlice(index=0, z_min=0.0, z_max=3.0, geometry=box(0, 0, 10, 10)),
        LayerSlice(index=1, z_min=3.0, z_max=6.0, geometry=box(0, 0, 10, 10)),
    ]
    parts = [
        Part(layer_index=0, cut_geometry=box(0, 0, 10, 10)),
        Part(layer_index=1, cut_geometry=box(0, 0, 10, 10)),
    ]

    mesh = build_stack_mesh(parts, layers)

    assert len(mesh.faces) > 0
    minz, maxz = mesh.bounds[0][2], mesh.bounds[1][2]
    assert minz == pytest.approx(0.0, abs=1e-6)
    assert maxz == pytest.approx(6.0, abs=1e-6)
    # Two 10x10x3 slabs stacked with no overlap -> total volume = 2 * 300.
    assert mesh.volume == pytest.approx(600.0, rel=1e-3)


def test_build_stack_mesh_respects_holes():
    ring = box(0, 0, 10, 10).difference(Point(5, 5).buffer(2))
    layers = [LayerSlice(index=0, z_min=0.0, z_max=2.0, geometry=ring)]
    parts = [Part(layer_index=0, cut_geometry=ring)]

    mesh = build_stack_mesh(parts, layers)

    solid_volume = 10.0 * 10.0 * 2.0
    assert mesh.volume < solid_volume
    assert mesh.volume == pytest.approx(ring.area * 2.0, rel=1e-2)


def test_build_stack_mesh_handles_multipolygon_and_uses_layer_z():
    from shapely.geometry import MultiPolygon

    multi = MultiPolygon([box(0, 0, 5, 5), box(20, 0, 25, 5)])
    layers = [LayerSlice(index=0, z_min=10.0, z_max=13.0, geometry=multi)]
    parts = [Part(layer_index=0, cut_geometry=multi)]

    mesh = build_stack_mesh(parts, layers)

    assert mesh.bounds[0][2] == pytest.approx(10.0, abs=1e-6)
    assert mesh.bounds[1][2] == pytest.approx(13.0, abs=1e-6)
    assert mesh.volume == pytest.approx(2 * 5.0 * 5.0 * 3.0, rel=1e-3)


def test_build_stack_mesh_skips_layers_with_no_matching_part():
    # Part references a layer index that isn't in `layers` -- should be
    # skipped rather than raising.
    parts = [Part(layer_index=5, cut_geometry=box(0, 0, 1, 1))]
    mesh = build_stack_mesh(parts, layers=[])
    assert len(mesh.faces) == 0


def test_build_stack_mesh_empty_input():
    mesh = build_stack_mesh([], [])
    assert len(mesh.faces) == 0
