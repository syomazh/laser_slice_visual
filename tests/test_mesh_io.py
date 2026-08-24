from __future__ import annotations

import os

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point

from laser_slice.config import Config
from laser_slice.mesh_io import load_mesh
from laser_slice.slicer import slice_mesh


def _notched_box() -> trimesh.Trimesh:
    """An asymmetric, non-rotationally-symmetric "step" solid.

    A 20x14x30 box with a corner notch cut out of the top third, so the
    cross section is a full 20x14 rectangle (area 280) for the bottom 2/3
    of the height and an L-shape (area 232) with a corner missing near
    (x=6, y=4) for the top 1/3. This distinctive, off-center feature makes
    it possible to detect axis mix-ups or mirroring in canonicalization.
    """
    box = trimesh.creation.box(extents=[20, 14, 30])
    notch = trimesh.creation.box(extents=[8, 6, 20])
    notch.apply_translation([6, 4, 15])
    mesh = box.difference(notch)
    assert mesh.is_watertight
    return mesh


def _export(mesh: trimesh.Trimesh, tmp_path, name: str) -> str:
    path = os.path.join(str(tmp_path), name)
    mesh.export(path)
    return path


def test_load_mesh_centers_xy_and_zeros_z(tmp_path):
    box = trimesh.creation.box(extents=[10, 20, 6])
    box.apply_translation([100, -50, 3])  # off-center, arbitrary Z offset
    path = _export(box, tmp_path, "box.obj")

    mesh = load_mesh(path, Config(slice_axis="z"))

    bounds = mesh.bounds
    assert bounds[0][2] == pytest.approx(0.0, abs=1e-9)
    assert bounds[1][2] == pytest.approx(6.0, abs=1e-9)
    center_x = (bounds[0][0] + bounds[1][0]) / 2.0
    center_y = (bounds[0][1] + bounds[1][1]) / 2.0
    assert center_x == pytest.approx(0.0, abs=1e-9)
    assert center_y == pytest.approx(0.0, abs=1e-9)


def test_load_mesh_model_height_mm_scales_z_extent_exactly(tmp_path):
    box = trimesh.creation.box(extents=[10, 20, 40])
    path = _export(box, tmp_path, "box.obj")

    mesh = load_mesh(path, Config(slice_axis="z", model_height_mm=100.0))

    z_extent = mesh.bounds[1][2] - mesh.bounds[0][2]
    assert z_extent == pytest.approx(100.0, abs=1e-9)
    # Uniform scale: X/Y should have scaled by the same factor (2.5x).
    x_extent = mesh.bounds[1][0] - mesh.bounds[0][0]
    assert x_extent == pytest.approx(25.0, abs=1e-9)


def test_load_mesh_scale_factor_multiplies_coordinates(tmp_path):
    box = trimesh.creation.box(extents=[10, 20, 40])
    path = _export(box, tmp_path, "box.obj")

    mesh = load_mesh(path, Config(slice_axis="z", scale_factor=2.5))

    z_extent = mesh.bounds[1][2] - mesh.bounds[0][2]
    x_extent = mesh.bounds[1][0] - mesh.bounds[0][0]
    assert z_extent == pytest.approx(100.0, abs=1e-9)
    assert x_extent == pytest.approx(25.0, abs=1e-9)


def test_load_mesh_no_scale_config_leaves_coordinates_unchanged(tmp_path):
    box = trimesh.creation.box(extents=[10, 20, 40])
    path = _export(box, tmp_path, "box.obj")

    mesh = load_mesh(path, Config(slice_axis="z"))

    z_extent = mesh.bounds[1][2] - mesh.bounds[0][2]
    x_extent = mesh.bounds[1][0] - mesh.bounds[0][0]
    assert z_extent == pytest.approx(40.0, abs=1e-9)
    assert x_extent == pytest.approx(10.0, abs=1e-9)


def test_load_mesh_warns_on_non_watertight_mesh(tmp_path, capsys):
    box = trimesh.creation.box(extents=[10, 10, 10])
    # Drop one quad (2 triangles) to open up the mesh.
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
    assert not open_mesh.is_watertight
    path = _export(open_mesh, tmp_path, "open.obj")

    load_mesh(path, Config(slice_axis="z"))

    captured = capsys.readouterr()
    assert "not watertight" in captured.err.lower()


def test_load_mesh_axis_canonicalization_matches_across_x_y_z(tmp_path):
    """Rotating the same solid so its "up" axis is X or Y, then loading it
    with the matching slice_axis, must reproduce the same canonical-frame
    geometry (and therefore the same per-layer slices) as loading the
    solid natively authored with Z as "up".

    The pre-rotations below are independently derived as the mathematical
    inverse of the documented canonicalization contract (config.slice_axis
    maps to +Z): i.e. they map the natural mesh's +Z onto +X (resp. +Y).
    If load_mesh's own rotation does not follow that exact convention, the
    three loads below will disagree.
    """
    natural = _notched_box()

    path_z = _export(natural, tmp_path, "z.obj")

    mesh_for_x = natural.copy()
    mesh_for_x.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    path_x = _export(mesh_for_x, tmp_path, "x.obj")

    mesh_for_y = natural.copy()
    mesh_for_y.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    path_y = _export(mesh_for_y, tmp_path, "y.obj")

    thickness = 5.0
    loaded = {
        "z": load_mesh(path_z, Config(material_thickness_mm=thickness, slice_axis="z")),
        "x": load_mesh(path_x, Config(material_thickness_mm=thickness, slice_axis="x")),
        "y": load_mesh(path_y, Config(material_thickness_mm=thickness, slice_axis="y")),
    }

    # All three should canonicalize to the identical bounding box.
    for axis, mesh in loaded.items():
        np.testing.assert_allclose(mesh.bounds, loaded["z"].bounds, atol=1e-6, err_msg=axis)

    layers = {
        axis: slice_mesh(mesh, Config(material_thickness_mm=thickness, slice_axis=axis))
        for axis, mesh in loaded.items()
    }

    assert len(layers["z"]) == len(layers["x"]) == len(layers["y"]) == 6

    notch_point = Point(6, 4)  # inside the notch void -> only "outside" material on top layers
    for i in range(6):
        areas = {}
        contains = {}
        for axis in ("z", "x", "y"):
            geom = layers[axis][i].geometry
            assert geom is not None, f"axis={axis} layer={i}"
            areas[axis] = geom.area
            contains[axis] = geom.contains(notch_point)

        # Areas must agree across the three axis conventions to within 0.1%.
        assert areas["x"] == pytest.approx(areas["z"], rel=1e-3), f"layer {i}: {areas}"
        assert areas["y"] == pytest.approx(areas["z"], rel=1e-3), f"layer {i}: {areas}"
        # The notch-point containment (a position-specific feature) must
        # also agree exactly across axis conventions at every layer.
        assert contains["x"] == contains["z"] == contains["y"], f"layer {i}: {contains}"

    # Sanity: the notch is actually present (bottom full rectangle, top L-shape).
    assert layers["z"][0].geometry.area == pytest.approx(280.0, rel=1e-6)
    assert layers["z"][5].geometry.area == pytest.approx(232.0, rel=1e-6)
