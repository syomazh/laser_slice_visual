from __future__ import annotations

import math

import pytest
import trimesh
from shapely.geometry import Point

from laser_slice.config import Config
from laser_slice.slicer import slice_mesh


def _canonicalize_z(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Shift a mesh so z_min == 0, mimicking mesh_io.load_mesh's precondition
    (slice_mesh assumes an already-canonicalized, Z-up, z_min=0 mesh)."""
    mesh = mesh.copy()
    mesh.apply_translation([0.0, 0.0, -mesh.bounds[0][2]])
    return mesh


def _notched_box() -> trimesh.Trimesh:
    """Same asymmetric "step" solid as in test_mesh_io.py: a 20x14x30 box
    with a corner notch removed from the top third, giving a full 20x14
    rectangle (area 280) for the bottom 2/3 of the height and an L-shape
    (area 232, with the notch void centered at model (x=6, y=4)) for the
    top 1/3. The cross section is constant (a true prism) within each of
    those two Z ranges, so any two layers within the same range must
    produce pixel-for-pixel identical (x, y) cross sections if -- and only
    if -- world X/Y stay consistent across cutting-plane origins."""
    box = trimesh.creation.box(extents=[20, 14, 30])
    notch = trimesh.creation.box(extents=[8, 6, 20])
    notch.apply_translation([6, 4, 15])
    mesh = box.difference(notch)
    assert mesh.is_watertight
    return _canonicalize_z(mesh)


def test_cylinder_ten_layers_uniform_area():
    cylinder = _canonicalize_z(trimesh.creation.cylinder(radius=10, height=50))
    config = Config(material_thickness_mm=5.0)

    layers = slice_mesh(cylinder, config)

    assert len(layers) == 10
    expected_area = math.pi * 10**2
    for layer in layers:
        assert layer.geometry is not None
        rel_error = abs(layer.geometry.area - expected_area) / expected_area
        assert rel_error < 0.05, f"layer {layer.index}: area={layer.geometry.area}"


def test_layer_z_bounds_are_contiguous_and_cover_full_height():
    cylinder = _canonicalize_z(trimesh.creation.cylinder(radius=10, height=50))
    config = Config(material_thickness_mm=5.0)

    layers = slice_mesh(cylinder, config)

    assert [l.index for l in layers] == list(range(10))
    for i, layer in enumerate(layers):
        assert layer.z_min == pytest.approx(i * 5.0)
        assert layer.z_max == pytest.approx((i + 1) * 5.0)
    assert layers[0].z_min == pytest.approx(0.0)
    assert layers[-1].z_max == pytest.approx(50.0)


def test_world_xy_consistent_across_different_height_layers():
    mesh = _notched_box()
    config = Config(material_thickness_mm=5.0)

    layers = slice_mesh(mesh, config)
    assert len(layers) == 6

    # A point inside the notch void: present as solid material in the
    # bottom (full-rectangle) layers, absent (cut away) in the top
    # (L-shape) layers.
    notch_point = Point(6, 4)

    bottom_areas = [layers[i].geometry.area for i in range(4)]
    for area in bottom_areas:
        assert area == pytest.approx(280.0, rel=1e-6)
        assert layers[0].geometry.contains(notch_point)

    # Two different-height layers within the L-shaped region (indices 4
    # and 5) must report the exact same containment result and area for
    # the same world (x, y) point/feature -- this is the world X/Y
    # consistency claim in slicer.py step 3. If to_planar()'s automatic
    # basis (or any other bug) let X/Y drift between cutting-plane
    # origins, this would fail.
    layer_a, layer_b = layers[4], layers[5]
    assert layer_a.geometry is not None and layer_b.geometry is not None
    assert layer_a.geometry.area == pytest.approx(layer_b.geometry.area, rel=1e-6)
    assert layer_a.geometry.area == pytest.approx(232.0, rel=1e-6)
    assert not layer_a.geometry.contains(notch_point)
    assert not layer_b.geometry.contains(notch_point)

    # The full polygons should coincide too (constant cross section / true
    # prism across that Z range), not just their areas. Vertex lists can
    # legitimately differ (mesh.section() may emit an extra collinear
    # vertex where the cutting plane crosses a side face's diagonal
    # triangulation edge), so compare via topological equality /
    # symmetric difference rather than exact vertex-for-vertex equality.
    assert layer_a.geometry.equals(layer_b.geometry)
    assert layer_a.geometry.symmetric_difference(layer_b.geometry).area == pytest.approx(0.0, abs=1e-9)
    assert layer_a.geometry.bounds == pytest.approx(layer_b.geometry.bounds, abs=1e-9)


def test_gap_in_mesh_produces_none_geometry_with_contiguous_indices():
    # Two disjoint boxes with a Z gap between them, concatenated into a
    # single mesh: the middle layer must have geometry=None, but layer
    # indices must remain contiguous (0, 1, 2) so downstream numbering
    # doesn't skip.
    box_a = trimesh.creation.box(extents=[10, 10, 10])
    box_a.apply_translation([0, 0, 5])  # z in [0, 10]
    box_b = trimesh.creation.box(extents=[10, 10, 10])
    box_b.apply_translation([0, 0, 25])  # z in [20, 30]
    mesh = trimesh.util.concatenate([box_a, box_b])

    config = Config(material_thickness_mm=10.0)
    layers = slice_mesh(mesh, config)

    assert [l.index for l in layers] == [0, 1, 2]
    assert layers[0].geometry is not None
    assert layers[0].geometry.area == pytest.approx(100.0, rel=1e-6)
    assert layers[1].geometry is None
    assert layers[2].geometry is not None
    assert layers[2].geometry.area == pytest.approx(100.0, rel=1e-6)


def test_clean_geometry_drops_tiny_slivers_and_keeps_significant_polygons():
    from shapely.geometry import MultiPolygon, Polygon

    from laser_slice.slicer import _clean_geometry

    big = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # area 100
    tiny = Polygon([(100, 100), (100.05, 100), (100, 100.05)])  # area ~0.00125 mm^2
    assert tiny.area < 0.01

    cleaned = _clean_geometry(MultiPolygon([big, tiny]))
    assert cleaned is not None
    assert cleaned.geom_type == "Polygon"
    assert cleaned.area == pytest.approx(100.0, rel=1e-6)


def test_clean_geometry_returns_none_when_everything_is_tiny_or_empty():
    from shapely.geometry import Polygon

    from laser_slice.slicer import _clean_geometry

    assert _clean_geometry(None) is None

    tiny = Polygon([(0, 0), (0.05, 0), (0, 0.05)])
    assert tiny.area < 0.01
    assert _clean_geometry(tiny) is None
