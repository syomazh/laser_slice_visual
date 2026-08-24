"""Visual sanity-check tools for a laser-slice job (mesh, layers, sheets).

Two entry points:

``render_overview`` -- always-works, non-interactive. Renders directly onto
a ``Figure``/``FigureCanvasAgg`` pair rather than going through pyplot's
global backend machinery, so it never needs a display, regardless of what
backend (if any) the rest of the process has configured. This is the
function ``cli.py`` is expected to call unconditionally.

``show_interactive`` -- best-effort desktop viewer built on ``pyplot`` with
whatever interactive backend happens to be available. It degrades
gracefully (prints a message, returns normally) if no display/backend is
available, instead of crashing; the caller is expected to have already
produced a ``render_overview`` PNG as a fallback.

Placement handling (used by ``show_interactive``'s sheet-layout panel)
follows the exact contract documented on
``laser_slice.geometry_types.Placement``: rotate a part's cut geometry (and
its engrave strokes, via the identical rigid transform) about the cut
geometry's own bounding-box center, then translate so the rotated bounding
box's min corner lands at (x_offset_mm, y_offset_mm). This is a plain
matplotlib preview in the CAD-style (Y-up) convention, so -- unlike
``svg_export.py`` -- no Y-flip is applied here.

Mesh Z convention: by the time a mesh reaches ``visualizer`` it has already
passed through ``mesh_io.load_mesh`` (and typically ``slicer.slice_mesh``),
both of which assume/guarantee the mesh is canonicalized to Z-up with
``z_min == 0`` (see ``slicer.py``'s module docstring). ``LayerSlice.z_min``/
``z_max`` are therefore always literal mesh Z coordinates, regardless of
``config.slice_axis`` (which only matters during the raw-mesh -> canonical
-mesh step in ``mesh_io``). Accordingly, the 3D views here always treat
world Z as the stacking axis.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import mpl_toolkits.mplot3d  # noqa: F401  (registers the '3d' projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient as _orient

from laser_slice.config import Config
from laser_slice.geometry_types import LayerSlice, Part, Placement, SheetLayout

__all__ = ["render_overview", "show_interactive", "render_stack_preview", "show_stack_preview"]

_UP_AXIS = 2  # world Z; see module docstring.

_MESH_FACECOLOR = (0.75, 0.75, 0.8, 0.35)
_MESH_EDGECOLOR = (0.25, 0.25, 0.25, 0.6)
_BOUNDARY_COLOR = "tab:blue"


# --------------------------------------------------------------------------
# mesh helpers
# --------------------------------------------------------------------------


def _mesh_triangles(mesh) -> np.ndarray:
    """Return an (n_faces, 3, 3) array of triangle vertices for `mesh`."""
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
    elif hasattr(mesh, "dump"):
        # e.g. a trimesh.Scene -- flatten to a single mesh.
        dumped = mesh.dump(concatenate=True)
        vertices = np.asarray(dumped.vertices)
        faces = np.asarray(dumped.faces)
    else:
        raise TypeError(f"Don't know how to extract triangles from mesh of type {type(mesh)!r}")
    return vertices[faces]


def _mesh_bounds(mesh) -> np.ndarray:
    """(2, 3) array [mins, maxs] for `mesh`."""
    if hasattr(mesh, "bounds") and mesh.bounds is not None:
        return np.asarray(mesh.bounds)
    tris = _mesh_triangles(mesh).reshape(-1, 3)
    return np.stack([tris.min(axis=0), tris.max(axis=0)])


def _boundary_quad(bounds: np.ndarray, up_idx: int, value: float) -> np.ndarray:
    """5 corner points (closed loop) of an axis-aligned quad at coordinate
    `up_idx` == `value`, spanning the other two axes' extent from `bounds`
    (a (2, 3) array: [mins, maxs])."""
    a_idx, b_idx = (i for i in range(3) if i != up_idx)
    lo_a, lo_b = bounds[0, a_idx], bounds[0, b_idx]
    hi_a, hi_b = bounds[1, a_idx], bounds[1, b_idx]
    corners_ab = [(lo_a, lo_b), (hi_a, lo_b), (hi_a, hi_b), (lo_a, hi_b), (lo_a, lo_b)]
    pts = np.zeros((5, 3))
    pts[:, up_idx] = value
    pts[:, a_idx] = [c[0] for c in corners_ab]
    pts[:, b_idx] = [c[1] for c in corners_ab]
    return pts


def _draw_mesh_surface(ax, mesh, bounds: np.ndarray) -> None:
    """Add the mesh itself (as a translucent Poly3DCollection) to a 3D axes
    and set sane limits/labels. Does not draw any layer-boundary markers."""
    tris = _mesh_triangles(mesh)
    coll = Poly3DCollection(tris, facecolor=_MESH_FACECOLOR, edgecolor=_MESH_EDGECOLOR, linewidths=0.15)
    ax.add_collection3d(coll)

    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    extents = bounds[1] - bounds[0]
    if hasattr(ax, "set_box_aspect") and np.all(extents > 0):
        try:
            ax.set_box_aspect(tuple(extents))
        except Exception:
            pass  # purely cosmetic; never fatal.


def _draw_stack_surface(ax, mesh, bounds: np.ndarray) -> None:
    """Like `_draw_mesh_surface`, but for a physically-reconstructed stack
    mesh (see `laser_slice.stack_preview.build_stack_mesh`): if the mesh
    carries per-face colors (set per layer, so adjacent physical layers are
    visibly distinguishable), use those instead of the single translucent
    tone used for the original smooth mesh."""
    tris = _mesh_triangles(mesh)
    face_colors = None
    visual = getattr(mesh, "visual", None)
    raw_colors = getattr(visual, "face_colors", None)
    if raw_colors is not None and len(raw_colors) == len(tris):
        face_colors = np.asarray(raw_colors, dtype=float) / 255.0

    coll = Poly3DCollection(
        tris,
        facecolor=face_colors if face_colors is not None else _MESH_FACECOLOR,
        edgecolor=(0.15, 0.1, 0.05, 0.5),
        linewidths=0.1,
    )
    ax.add_collection3d(coll)

    ax.set_xlim(bounds[0, 0], bounds[1, 0])
    ax.set_ylim(bounds[0, 1], bounds[1, 1])
    ax.set_zlim(bounds[0, 2], bounds[1, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    extents = bounds[1] - bounds[0]
    if hasattr(ax, "set_box_aspect") and np.all(extents > 0):
        try:
            ax.set_box_aspect(tuple(extents))
        except Exception:
            pass  # purely cosmetic; never fatal.


def _draw_all_boundaries(ax, bounds: np.ndarray, layers: list[LayerSlice], alpha: float = 0.10) -> None:
    """Draw a translucent quad + outline at every distinct layer boundary
    Z value (used by render_overview only)."""
    boundary_values = sorted({round(v, 9) for layer in layers for v in (layer.z_min, layer.z_max)})
    for value in boundary_values:
        pts = _boundary_quad(bounds, _UP_AXIS, value)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=_BOUNDARY_COLOR, linewidth=0.8, alpha=0.8)
        quad = Poly3DCollection([pts[:-1]], facecolor=_BOUNDARY_COLOR, alpha=alpha)
        ax.add_collection3d(quad)


# --------------------------------------------------------------------------
# 2D polygon -> matplotlib Path helpers
# --------------------------------------------------------------------------


def _polygon_to_path(polygon) -> Path | None:
    """Build a matplotlib compound Path for a single shapely Polygon whose
    holes render as holes.

    Rings are re-oriented (exterior CCW, holes CW) via
    ``shapely.geometry.polygon.orient``. A compound path built from rings of
    opposite winding fills identically under Agg's nonzero-winding-rule
    renderer as it would under an explicit even-odd rule (for the simple,
    non-self-intersecting rings shapely produces), so this reliably renders
    holes as holes regardless of the winding of the input polygon.
    """
    if polygon.is_empty:
        return None
    polygon = _orient(polygon, sign=1.0)
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in (polygon.exterior, *polygon.interiors):
        coords = list(ring.coords)
        if len(coords) < 2:
            continue
        vertices.extend(coords)
        codes.append(Path.MOVETO)
        codes.extend([Path.LINETO] * (len(coords) - 2))
        codes.append(Path.CLOSEPOLY)
    if not vertices:
        return None
    return Path(vertices, codes)


def _geometry_to_path(geometry: BaseGeometry | None) -> Path | None:
    """Build a matplotlib compound Path for a shapely Polygon, MultiPolygon,
    or GeometryCollection-of-polygons (or None/empty -> None)."""
    if geometry is None or geometry.is_empty:
        return None
    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        return _polygon_to_path(geometry)
    if geom_type in ("MultiPolygon", "GeometryCollection"):
        paths = [
            p
            for g in geometry.geoms
            if g.geom_type == "Polygon" and not g.is_empty
            for p in [_polygon_to_path(g)]
            if p is not None
        ]
        if not paths:
            return None
        return Path.make_compound_path(*paths)
    return None


def _draw_silhouette(ax, geometry: BaseGeometry | None, facecolor="0.8", edgecolor="black", linewidth=0.8) -> None:
    """Draw one layer's 2D silhouette (with holes) into `ax`."""
    path = _geometry_to_path(geometry)
    if path is None:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
        return
    patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(patch)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")


# --------------------------------------------------------------------------
# Placement contract (see geometry_types.Placement docstring)
# --------------------------------------------------------------------------


def _apply_placement(part: Part, placement: Placement):
    """Materialize a Placement: rotate ``part.cut_geometry`` about its own
    bounding-box center, then translate its bounds' min corner to
    ``(placement.x_offset_mm, placement.y_offset_mm)``; apply the identical
    rigid transform to every engrave stroke so it stays attached to the
    part. Returns ``(placed_geometry, placed_stroke_point_lists)``, both in
    the CAD-style (Y-up) sheet convention -- no SVG Y-flip here.
    """
    import shapely.affinity as affinity

    minx0, miny0, maxx0, maxy0 = part.cut_geometry.bounds
    center = ((minx0 + maxx0) / 2.0, (miny0 + maxy0) / 2.0)

    geom = affinity.rotate(part.cut_geometry, placement.rotation_deg, origin="center")
    minx, miny, _, _ = geom.bounds
    xoff = placement.x_offset_mm - minx
    yoff = placement.y_offset_mm - miny
    geom = affinity.translate(geom, xoff=xoff, yoff=yoff)

    theta = math.radians(placement.rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = center

    strokes: list[list[tuple[float, float]]] = []
    for glyph in part.engrave_strokes:
        pts = []
        for x, y in glyph.points:
            rx, ry = x - cx, y - cy
            rxr = rx * cos_t - ry * sin_t
            ryr = rx * sin_t + ry * cos_t
            pts.append((rxr + cx + xoff, ryr + cy + yoff))
        strokes.append(pts)
    return geom, strokes


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def render_overview(
    mesh, layers: list[LayerSlice], sheets: list[SheetLayout], config: Config, out_path: str
) -> None:
    """Render a static, non-interactive sanity-check figure to `out_path`.

    Always works without a display: this builds a ``Figure`` and binds it
    to a ``FigureCanvasAgg`` directly (rather than going through pyplot's
    global, possibly-GUI backend), which is what "defaults to Agg" means in
    practice for a function that only ever calls ``savefig``. It never
    touches pyplot's global state, so it's also safe to call from the same
    process as ``show_interactive``.

    Layout: a 3D view of `mesh` (translucent Poly3DCollection surface) with
    every layer boundary Z drawn as a translucent horizontal plane + outline,
    plus a grid of thumbnails -- one per layer -- of that layer's 2D
    silhouette (holes rendered as holes), labeled with its layer index.

    `sheets` is accepted for interface symmetry with `show_interactive` but
    is not itself rendered here; this overview is specifically a sanity
    check of the mesh -> layer slicing step (nesting has its own SVG output
    to review).
    """
    n_layers = len(layers)
    ncols_thumb = min(4, max(1, n_layers))
    nrows_thumb = max(1, math.ceil(n_layers / ncols_thumb)) if n_layers else 1

    fig_w = max(8.0, (ncols_thumb + 2) * 2.4)
    fig_h = max(4.5, nrows_thumb * 2.4 + 1.0)
    fig = Figure(figsize=(fig_w, fig_h))
    FigureCanvasAgg(fig)  # binds a non-interactive Agg canvas; no display needed.

    gs = fig.add_gridspec(nrows_thumb, ncols_thumb + 2, wspace=0.4, hspace=0.5)

    bounds = _mesh_bounds(mesh)
    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    _draw_mesh_surface(ax3d, mesh, bounds)
    _draw_all_boundaries(ax3d, bounds, layers)
    ax3d.set_title(f"mesh + {n_layers} layer(s)")

    for i, layer in enumerate(layers):
        row, col = divmod(i, ncols_thumb)
        ax = fig.add_subplot(gs[row, 2 + col])
        _draw_silhouette(ax, layer.geometry)
        ax.set_title(f"layer {layer.index}", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle("laser_slice overview")
    fig.savefig(out_path, dpi=150)


def show_interactive(mesh, layers: list[LayerSlice], sheets: list[SheetLayout], config: Config) -> None:
    """Best-effort interactive viewer for local desktop use.

    Three linked subplots -- (1) the mesh in 3D with a translucent plane at
    the selected layer's z_mid, (2) that layer's 2D silhouette plus its
    engrave strokes, (3) the sheet layout containing that layer with the
    current part outlined -- driven by a Slider that scrubs the layer index.
    The Up and Down arrow keys advance to the next and previous layer.

    If no display/interactive backend is available, this prints a message
    and returns normally instead of raising (the caller is expected to have
    already produced a fallback image via `render_overview`).
    """
    if not layers:
        print("show_interactive: no layers to display.")
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        # layer_index -> (Part, SheetLayout) it ended up nested on, if any.
        part_by_layer: dict[int, Part] = {}
        sheet_by_layer: dict[int, SheetLayout] = {}
        for sheet in sheets:
            for placement in sheet.placements:
                part_by_layer[placement.part.layer_index] = placement.part
                sheet_by_layer[placement.part.layer_index] = sheet

        bounds = _mesh_bounds(mesh)

        fig = plt.figure(figsize=(15, 5.5))
        gs = fig.add_gridspec(1, 3, bottom=0.22, wspace=0.35)
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax2d = fig.add_subplot(gs[0, 1])
        ax_sheet = fig.add_subplot(gs[0, 2])

        _draw_mesh_surface(ax3d, mesh, bounds)

        state = {"plane": None}

        def draw_layer(idx: int) -> None:
            layer = layers[idx]

            # --- (1) 3D mesh + translucent current-layer plane ---
            if state["plane"] is not None:
                state["plane"].remove()
                state["plane"] = None
            pts = _boundary_quad(bounds, _UP_AXIS, layer.z_mid)
            quad = Poly3DCollection([pts[:-1]], facecolor="tab:orange", alpha=0.35)
            ax3d.add_collection3d(quad)
            state["plane"] = quad
            ax3d.set_title(f"layer {layer.index}  z_mid={layer.z_mid:.2f}")

            # --- (2) layer silhouette + engrave strokes ---
            ax2d.clear()
            _draw_silhouette(ax2d, layer.geometry, facecolor="0.85", edgecolor="black")
            part = part_by_layer.get(layer.index)
            if part is not None:
                for glyph in part.engrave_strokes:
                    if len(glyph.points) < 2:
                        continue
                    xs, ys = zip(*glyph.points)
                    ax2d.plot(xs, ys, color="red", linewidth=1.0)
            ax2d.set_title(f"layer {layer.index}: silhouette + engraving")
            ax2d.set_aspect("equal", adjustable="datalim")

            # --- (3) sheet layout, current part highlighted ---
            ax_sheet.clear()
            sheet = sheet_by_layer.get(layer.index)
            if sheet is None:
                ax_sheet.text(0.5, 0.5, "not nested", ha="center", va="center", transform=ax_sheet.transAxes)
            else:
                ax_sheet.add_patch(
                    plt.Rectangle(
                        (0, 0), sheet.width_mm, sheet.height_mm, fill=False, edgecolor="black", linewidth=1.0
                    )
                )
                for placement in sheet.placements:
                    is_current = placement.part.layer_index == layer.index
                    geom, strokes = _apply_placement(placement.part, placement)
                    path = _geometry_to_path(geom)
                    if path is not None:
                        patch = PathPatch(
                            path,
                            facecolor="tab:orange" if is_current else "0.9",
                            edgecolor="red" if is_current else "black",
                            linewidth=1.6 if is_current else 0.6,
                            zorder=2 if is_current else 1,
                        )
                        ax_sheet.add_patch(patch)
                    if is_current:
                        for stroke in strokes:
                            if len(stroke) < 2:
                                continue
                            xs, ys = zip(*stroke)
                            ax_sheet.plot(xs, ys, color="darkred", linewidth=0.8, zorder=3)
                ax_sheet.set_xlim(-5, sheet.width_mm + 5)
                ax_sheet.set_ylim(-5, sheet.height_mm + 5)
                ax_sheet.set_title(f"sheet {sheet.sheet_index}")
            ax_sheet.set_aspect("equal", adjustable="box")

            fig.canvas.draw_idle()

        draw_layer(0)

        slider = None
        key_press_handler = None
        key_press_connection = None
        if len(layers) > 1:
            slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
            slider = Slider(slider_ax, "layer (Up/Down keys)", 0, len(layers) - 1, valinit=0, valstep=1)
            slider.on_changed(lambda val: draw_layer(int(val)))

            def on_key_press(event) -> None:
                step = {"up": 1, "down": -1}.get(event.key)
                if step is None:
                    return
                current_index = int(slider.val)
                next_index = max(0, min(len(layers) - 1, current_index + step))
                if next_index != current_index:
                    slider.set_val(next_index)

            key_press_handler = on_key_press
            key_press_connection = fig.canvas.mpl_connect("key_press_event", on_key_press)
        # Keep references alive on the figure (avoids GC of the widget/state).
        fig._laser_slice_slider = slider
        fig._laser_slice_state = state
        fig._laser_slice_key_press_handler = key_press_handler
        fig._laser_slice_key_press_connection = key_press_connection

        plt.show()
    except Exception as exc:
        print(
            "show_interactive: could not display the interactive viewer "
            f"({type(exc).__name__}: {exc}). An overview image should already "
            "have been saved by render_overview()."
        )


def render_stack_preview(stack_mesh, out_path: str) -> None:
    """Render a static, non-interactive 3D view of the physically
    reconstructed stack (see `laser_slice.stack_preview.build_stack_mesh`)
    to `out_path`. This shows what the glued-up physical object will
    actually look like -- including the visible layer "stairstep" -- unlike
    a smooth render of the original mesh.

    Never requires a display, for the same reason as `render_overview`
    (builds a Figure bound to a non-interactive Agg canvas directly).
    """
    fig = Figure(figsize=(7.0, 7.0))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    if len(stack_mesh.faces) == 0:
        ax.text2D(
            0.5, 0.5, "no cuttable geometry to preview",
            ha="center", va="center", transform=ax.transAxes,
        )
    else:
        bounds = _mesh_bounds(stack_mesh)
        _draw_stack_surface(ax, stack_mesh, bounds)

    ax.set_title("laser_slice stack preview (physical reconstruction)")
    ax.view_init(elev=18, azim=-60)
    fig.savefig(out_path, dpi=150)


def show_stack_preview(stack_mesh) -> None:
    """Best-effort interactive 3D viewer for the physically reconstructed
    stack. Freely rotate/zoom with the mouse to inspect the assembled
    object from any angle.

    Degrades gracefully (prints a message, returns normally) if no
    display/backend is available, the same as `show_interactive`; the
    caller is expected to have already produced a `render_stack_preview`
    PNG as a fallback.
    """
    if len(stack_mesh.faces) == 0:
        print("show_stack_preview: no cuttable geometry to preview.")
        return

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        bounds = _mesh_bounds(stack_mesh)
        _draw_stack_surface(ax, stack_mesh, bounds)
        ax.set_title("laser_slice stack preview -- drag to rotate")
        ax.view_init(elev=18, azim=-60)
        plt.show()
    except Exception as exc:
        print(
            "show_stack_preview: could not display the interactive viewer "
            f"({type(exc).__name__}: {exc}). A stack_preview.png should already "
            "have been saved by render_stack_preview()."
        )
