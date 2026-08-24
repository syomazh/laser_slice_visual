from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from shapely.geometry import box
import trimesh

from laser_slice.config import Config
from laser_slice.geometry_types import LayerSlice
from laser_slice.visualizer import show_interactive


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
    finally:
        plt.close("all")
