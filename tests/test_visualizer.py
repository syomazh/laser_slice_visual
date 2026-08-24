from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
import pytest
from shapely.geometry import box
import trimesh

from laser_slice.config import Config
from laser_slice.geometry_types import LayerSlice
from laser_slice.visualizer import show_interactive, show_stack_preview


def _click_button(button) -> None:
    canvas = button.ax.figure.canvas
    canvas.draw()
    x, y = button.ax.transAxes.transform((0.5, 0.5))
    for event_name in ("button_press_event", "button_release_event"):
        event = MouseEvent(event_name, canvas, x, y, button=MouseButton.LEFT)
        canvas.callbacks.process(event_name, event)


def test_interactive_viewer_arrow_keys_change_layer_and_sync_slider(monkeypatch):
    plt.switch_backend("Agg")
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 3.0))
    mesh.apply_translation((0.0, 0.0, 1.5))
    layers = [
        LayerSlice(index=10, z_min=0.0, z_max=1.0, geometry=box(-1.0, -1.0, 1.0, 1.0)),
        LayerSlice(index=20, z_min=1.0, z_max=2.0, geometry=box(-0.8, -0.8, 0.8, 0.8)),
        LayerSlice(index=30, z_min=2.0, z_max=3.0, geometry=box(-0.5, -0.5, 0.5, 0.5)),
    ]

    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        show_interactive(mesh, layers, sheets=[], config=Config())
        fig = plt.gcf()
        slider = fig._laser_slice_slider

        def press(key: str) -> None:
            event = KeyEvent("key_press_event", fig.canvas, key=key)
            fig.canvas.callbacks.process("key_press_event", event)

        assert slider.val == 0
        assert "layer 10" in fig.axes[0].get_title()

        press("down")
        assert slider.val == 0

        press("up")
        assert slider.val == 1
        assert "layer 20" in fig.axes[0].get_title()

        press("up")
        press("up")
        assert slider.val == 2
        assert "layer 30" in fig.axes[0].get_title()

        press("down")
        assert slider.val == 1
        assert "layer 20" in fig.axes[0].get_title()

        ax3d = fig._laser_slice_view_axes
        buttons = fig._laser_slice_view_buttons
        assert set(buttons) == {"rotate_left", "rotate_right", "front", "side", "top", "isometric"}

        starting_elevation, starting_azimuth = ax3d.elev, ax3d.azim
        _click_button(buttons["rotate_left"])
        assert ax3d.elev == pytest.approx(starting_elevation)
        assert ax3d.azim == pytest.approx(starting_azimuth - 15.0)
        _click_button(buttons["rotate_right"])
        assert ax3d.azim == pytest.approx(starting_azimuth)

        for name, expected in {
            "front": (0.0, -90.0),
            "side": (0.0, 0.0),
            "top": (90.0, -90.0),
            "isometric": (30.0, -60.0),
        }.items():
            _click_button(buttons[name])
            assert (ax3d.elev, ax3d.azim) == pytest.approx(expected)
    finally:
        plt.close("all")


def test_stack_preview_has_rotation_and_preset_controls(monkeypatch):
    plt.switch_backend("Agg")
    stack_mesh = trimesh.creation.box(extents=(2.0, 2.0, 3.0))
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        show_stack_preview(stack_mesh)
        fig = plt.gcf()
        buttons = fig._laser_slice_view_buttons

        assert set(buttons) == {"rotate_left", "rotate_right", "front", "side", "top", "isometric"}
        _click_button(buttons["isometric"])
        assert (fig._laser_slice_view_axes.elev, fig._laser_slice_view_axes.azim) == pytest.approx((30.0, -60.0))
    finally:
        plt.close("all")
