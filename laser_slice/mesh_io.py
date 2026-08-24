"""Mesh loading and canonicalization.

``load_mesh`` is the single entry point downstream stages rely on: it loads
an arbitrary 3D model file, reorients it so the model's "up" axis
(``config.slice_axis``) becomes world +Z, scales it into millimeters, and
translates it so the mesh sits with its minimum Z at exactly 0 and its XY
bounding-box center at the origin (x=0, y=0).

IMPORTANT: ``config.registration_points``, if the user sets them explicitly
(rather than letting them be auto-computed later), must be specified as
(x, y) coordinates in this exact final canonicalized/centered frame -- i.e.
the frame of the mesh *after* ``load_mesh`` has returned it, not the raw
frame of the source file.
"""
from __future__ import annotations

import sys

import numpy as np
import trimesh

from laser_slice.config import Config

# For each non-"z" slice_axis: (rotation direction vector, angle in radians)
# such that applying trimesh.transformations.rotation_matrix(angle, direction)
# maps the world +<axis> direction onto world +Z.
_AXIS_ROTATIONS: dict[str, tuple[list[float], float]] = {
    "x": ([0.0, 1.0, 0.0], -np.pi / 2),  # rotate -90 deg about Y: +X -> +Z
    "y": ([1.0, 0.0, 0.0], np.pi / 2),  # rotate +90 deg about X: +Y -> +Z
}


def load_mesh(path: str, config: Config) -> trimesh.Trimesh:
    """Load a mesh file and canonicalize it into laser_slice's working frame.

    Steps:
      1. Load (multi-object files are merged into a single Trimesh).
      2. Rotate so config.slice_axis becomes world +Z.
      3. Scale into millimeters per config.model_height_mm / scale_factor.
      4. Translate so z_min == 0 and the XY bounding-box is centered at
         (x=0, y=0).
      5. Warn (to stderr) if the resulting mesh is not watertight.
    """
    mesh = trimesh.load(path, force="mesh")

    axis = config.slice_axis.lower()
    axis_index = config.slice_axis_index()
    original_extent = float(mesh.bounds[1][axis_index] - mesh.bounds[0][axis_index])

    # --- 2. canonicalize orientation: config.slice_axis -> world +Z ---
    if axis != "z":
        direction, angle = _AXIS_ROTATIONS[axis]
        transform = trimesh.transformations.rotation_matrix(angle, direction)
        mesh.apply_transform(transform)

    new_z_extent = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    assert np.isclose(new_z_extent, original_extent, rtol=1e-5, atol=1e-6), (
        "axis canonicalization self-check failed: expected Z extent "
        f"{original_extent!r} (original extent along '{axis}'), got {new_z_extent!r}"
    )

    # --- 3. scale into millimeters ---
    if config.model_height_mm is not None:
        z_extent = float(mesh.bounds[1][2] - mesh.bounds[0][2])
        if z_extent > 0:
            mesh.apply_scale(config.model_height_mm / z_extent)
    elif config.scale_factor is not None:
        mesh.apply_scale(config.scale_factor)
    # else: coordinates are assumed to already be in millimeters.

    # --- 4. translate: z_min -> 0, XY bbox center -> (0, 0) ---
    bounds = mesh.bounds
    center_x = (bounds[0][0] + bounds[1][0]) / 2.0
    center_y = (bounds[0][1] + bounds[1][1]) / 2.0
    z_min = bounds[0][2]
    mesh.apply_translation([-center_x, -center_y, -z_min])

    # --- 5. watertightness warning ---
    if not mesh.is_watertight:
        print(
            f"Warning: mesh loaded from {path!r} is not watertight; "
            "slicing may produce open or inconsistent contours.",
            file=sys.stderr,
        )

    return mesh
