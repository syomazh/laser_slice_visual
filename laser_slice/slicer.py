"""Horizontal slicing of a canonicalized mesh into per-layer 2D cross sections.

``slice_mesh`` assumes the mesh has already been canonicalized by
``mesh_io.load_mesh`` (Z-up, ``z_min == 0``). It cuts the mesh into
horizontal slabs of thickness ``config.material_thickness_mm`` and computes
an exact-projection 2D cross section (shapely Polygon/MultiPolygon) at the
mid-height of each slab.

Cross sections are deliberately computed via ``mesh.section(...).discrete``
(closed 3D loops) projected to 2D by simply dropping Z, rather than via
``Path3D.to_planar()``. ``to_planar()`` picks an arbitrary 2D basis for the
cutting plane which is not guaranteed to stay aligned with world X/Y across
different plane origins -- since the cutting plane normal here is always
exactly [0, 0, 1], dropping Z is an exact (not approximate) projection, and
it guarantees identical (x, y) world coordinates across every layer. This is
required for registration holes (added downstream) to line up between
layers.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from laser_slice.config import Config
from laser_slice.geometry_types import LayerSlice

_MIN_POLYGON_AREA_MM2 = 0.01


def slice_mesh(mesh: trimesh.Trimesh, config: Config) -> list[LayerSlice]:
    """Slice a canonicalized (Z-up, z_min=0) mesh into LayerSlice objects.

    Returns one LayerSlice per layer index, for ALL layers spanning the
    mesh's Z extent -- including layers whose geometry is None (no
    intersection with the mesh) -- so downstream layer numbering stays
    contiguous.
    """
    total_height = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    thickness = config.material_thickness_mm
    n_layers = max(1, math.ceil(total_height / thickness))

    layers: list[LayerSlice] = []
    for i in range(n_layers):
        z_min = i * thickness
        z_max = min((i + 1) * thickness, total_height)
        z_mid = (z_min + z_max) / 2.0

        geometry = _cross_section(mesh, z_mid)
        geometry = _clean_geometry(geometry)

        layers.append(LayerSlice(index=i, z_min=z_min, z_max=z_max, geometry=geometry))

    return layers


def _cross_section(mesh: trimesh.Trimesh, z_mid: float) -> BaseGeometry | None:
    """Exact-projection 2D cross section of ``mesh`` at height ``z_mid``."""
    section = mesh.section(plane_origin=[0, 0, z_mid], plane_normal=[0, 0, 1])
    if section is None:
        return None

    loops_3d = [loop for loop in section.discrete if len(loop) >= 3]
    if not loops_3d:
        return None

    # Pure drop-Z projection: exact because the plane normal is always +Z,
    # so world X/Y are identical to the plane's local X/Y at every height.
    loops_2d = [loop[:, :2] for loop in loops_3d]

    # trimesh.load_path expects either a single connected polyline
    # ((n, 2) points) or explicit line segments ((n, 2, 2) pairs); a list of
    # multiple closed loops must be flattened into segments so trimesh's own
    # entity/containment logic (used by polygons_full below) can tell loops
    # apart and resolve holes correctly.
    segments = np.vstack(
        [np.stack([loop[:-1], loop[1:]], axis=1) for loop in loops_2d]
    )
    path2d = trimesh.load_path(segments)

    polys = path2d.polygons_full
    if not polys:
        return None
    return unary_union(polys)


def _clean_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
    """Fix self-intersections via buffer(0) and drop slivers < 0.01 mm^2."""
    if geometry is None:
        return None

    cleaned = geometry.buffer(0)
    if cleaned.is_empty:
        return None

    if cleaned.geom_type == "Polygon":
        candidates = [cleaned]
    elif cleaned.geom_type == "MultiPolygon":
        candidates = list(cleaned.geoms)
    else:
        # GeometryCollection or similar (rare, from a degenerate buffer(0));
        # keep only the polygonal parts.
        candidates = [g for g in getattr(cleaned, "geoms", [cleaned]) if g.geom_type == "Polygon"]

    kept = [p for p in candidates if p.area >= _MIN_POLYGON_AREA_MM2]
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return MultiPolygon(kept)
