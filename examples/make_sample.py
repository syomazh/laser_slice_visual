"""Build a small sculptural test mesh and export it to examples/sample.obj.

This is a rough "figurine" -- a cylindrical torso, a spherical head, two
cylindrical legs, and two diagonal cylindrical arms -- assembled with
trimesh.creation primitives and combined into a single watertight solid via
a manifold3d-backed boolean union (trimesh.boolean.union(..., engine=
"manifold")). No Blender or external CAD tool is required.

The arms are deliberately routed diagonally: each arm's "shoulder" end sits
well inside the torso (so it fuses into the torso volume, keeping the whole
mesh a single watertight solid) while its "hand" end reaches far outside the
torso's radius at a *lower* Z height. Consequently, a horizontal slice near
the hand end shows the arm as a contour completely separate from the torso's
contour -- i.e. the silhouette has multiple disjoint regions at that height
-- which is exactly the multi-contour-per-layer case laser_slice.slicer must
handle correctly (see the module docstring in laser_slice/slicer.py).

Run directly to (re)generate examples/sample.obj:

    ../.venv/Scripts/python.exe examples/make_sample.py
"""
from __future__ import annotations

import os

import trimesh
import trimesh.creation


def build_sample_mesh() -> trimesh.Trimesh:
    # --- torso: a vertical cylinder from z=0 to z=80 ---
    body_radius = 20.0
    body_height = 80.0
    body = trimesh.creation.cylinder(radius=body_radius, height=body_height, sections=48)
    body.apply_translation([0.0, 0.0, body_height / 2.0])

    # --- head: a sphere sitting on top of, and overlapping into, the torso ---
    head_radius = 15.0
    head = trimesh.creation.icosphere(subdivisions=3, radius=head_radius)
    head.apply_translation([0.0, 0.0, body_height + head_radius * 0.4])

    # --- legs: two vertical cylinders overlapping into the torso's bottom ---
    leg_radius = 7.0
    leg_height = 34.0
    leg_x_offset = 10.0
    leg_overlap = 4.0
    legs = []
    for sign in (-1.0, 1.0):
        leg = trimesh.creation.cylinder(radius=leg_radius, height=leg_height, sections=32)
        leg.apply_translation([sign * leg_x_offset, 0.0, -leg_height / 2.0 + leg_overlap])
        legs.append(leg)

    # --- arms: two diagonal cylinders, shoulder embedded in the torso, hand
    # reaching far outside it at a lower Z -- see module docstring. ---
    arm_radius = 6.0
    arms = []
    for sign in (-1.0, 1.0):
        shoulder = [sign * 10.0, 0.0, 68.0]  # inside the torso (radius 20) -> fuses on union
        hand = [sign * 55.0, 0.0, 42.0]  # far outside the torso -> separate contour
        arm = trimesh.creation.cylinder(radius=arm_radius, sections=24, segment=[shoulder, hand])
        arms.append(arm)

    parts = [body, head, *legs, *arms]
    mesh = trimesh.boolean.union(parts, engine="manifold")
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def main() -> None:
    mesh = build_sample_mesh()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "sample.obj")
    mesh.export(out_path)

    print(f"wrote {out_path}")
    print(f"  watertight: {mesh.is_watertight}")
    print(f"  vertices:   {len(mesh.vertices)}")
    print(f"  faces:      {len(mesh.faces)}")
    print(f"  bounds:     {mesh.bounds.tolist()}")


if __name__ == "__main__":
    main()
