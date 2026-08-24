"""Reconstruct the physical stacked object from final cut layer geometry.

Extrudes each Part's already-cut (kerf-compensated, registration-holes-and-
all) 2D geometry to its real slab thickness and stacks the slabs at their
true Z position. This is what the glued-up physical object will actually
look like -- including the visible "stairstep" from approximating the
model with flat horizontal slabs -- unlike a smooth render of the original
mesh, which hides that entirely.

Layers are stacked as independent, non-unioned solids (one extruded prism
per Part, just translated into place and concatenated), matching physical
reality: each layer is a separate piece of material glued to its
neighbors, not a single fused solid.
"""
from __future__ import annotations

import numpy as np
import trimesh
import trimesh.creation

from laser_slice.geometry_types import LayerSlice, Part

# Two alternating basswood-ish tones so adjacent physical layers are visibly
# distinguishable in the preview, the way real glued veneer layers are.
_LAYER_COLORS = [
    (222, 184, 135, 255),  # burlywood
    (205, 170, 125, 255),  # a shade darker
]


def _iter_polygons(geometry):
    """Yield the individual (non-empty) shapely Polygons making up
    `geometry`, which may be a Polygon, MultiPolygon, or None/empty."""
    if geometry is None or geometry.is_empty:
        return
    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        yield geometry
    elif geom_type in ("MultiPolygon", "GeometryCollection"):
        for sub in geometry.geoms:
            if sub.geom_type == "Polygon" and not sub.is_empty:
                yield sub


def build_stack_mesh(parts: list[Part], layers: list[LayerSlice]) -> trimesh.Trimesh:
    """Build a single (non-manifold-between-layers) Trimesh representing the
    physically assembled stack: every part extruded to its layer's real
    thickness (z_max - z_min, which may be less than the nominal material
    thickness for a final partial layer) and positioned at that layer's
    z_min.

    Returns an empty Trimesh if there is nothing to build (no parts, or
    every part's geometry failed to triangulate/extrude).
    """
    layers_by_index = {layer.index: layer for layer in layers}

    slabs = []
    for part in parts:
        layer = layers_by_index.get(part.layer_index)
        if layer is None:
            continue
        thickness = layer.z_max - layer.z_min
        if thickness <= 0:
            continue

        for polygon in _iter_polygons(part.cut_geometry):
            if polygon.area <= 0:
                continue
            try:
                slab = trimesh.creation.extrude_polygon(polygon, height=thickness)
            except Exception:
                # A degenerate/self-intersecting ring occasionally fails to
                # triangulate; skip just that piece rather than the whole
                # preview.
                continue
            slab.apply_translation([0.0, 0.0, layer.z_min])
            color = _LAYER_COLORS[layer.index % len(_LAYER_COLORS)]
            slab.visual.face_colors = np.tile(color, (len(slab.faces), 1))
            slabs.append(slab)

    if not slabs:
        return trimesh.Trimesh()

    return trimesh.util.concatenate(slabs)
