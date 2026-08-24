"""Tests for laser_slice.nesting.nest_parts.

Validates the FFDH shelf-packing behavior and, independently, that every
Placement it returns actually obeys the coordinate contract documented on
laser_slice.geometry_types.Placement (rotate-about-bbox-center, then
translate the rotated bbox's min corner to (x_offset_mm, y_offset_mm)).
"""
from __future__ import annotations

import pytest
import shapely.affinity as affinity
from shapely.geometry import box

from laser_slice.config import Config
from laser_slice.geometry_types import Part, Placement
from laser_slice.nesting import nest_parts

# A hardcoded, varied set of rectangular part sizes (width_mm, height_mm),
# including several elongated pieces that benefit from rotation. 24 parts
# total, sized so a 300x300mm sheet forces multiple sheets to be used.
SIZES: list[tuple[float, float]] = [
    (120, 90),
    (110, 95),
    (100, 100),
    (95, 60),
    (90, 90),
    (85, 40),
    (80, 80),
    (75, 50),
    (70, 70),
    (65, 130),
    (60, 60),
    (55, 45),
    (50, 200),
    (45, 45),
    (40, 150),
    (38, 38),
    (35, 90),
    (30, 30),
    (28, 28),
    (150, 20),
    (200, 15),
    (25, 60),
    (20, 20),
    (15, 15),
]

EPS = 1e-6


def _build_parts(sizes: list[tuple[float, float]]) -> list[Part]:
    return [
        Part(layer_index=i, cut_geometry=box(0.0, 0.0, w, h))
        for i, (w, h) in enumerate(sizes)
    ]


def _materialize_bounds(placement: Placement) -> tuple[float, float, float, float]:
    """Reproduce the exact recipe from the Placement docstring and return the
    resulting geometry's bounds, independently of anything nesting.py assumed
    internally."""
    geom = affinity.rotate(placement.part.cut_geometry, placement.rotation_deg, origin="center")
    minx, miny, _, _ = geom.bounds
    geom = affinity.translate(
        geom, xoff=placement.x_offset_mm - minx, yoff=placement.y_offset_mm - miny
    )
    return geom.bounds


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 < bx1 - EPS) and (bx0 < ax1 - EPS) and (ay0 < by1 - EPS) and (by0 < ay1 - EPS)


def _validate_layouts(layouts, parts, config: Config) -> None:
    usable_minx = config.sheet_margin_mm
    usable_miny = config.sheet_margin_mm
    usable_maxx = config.sheet_width_mm - config.sheet_margin_mm
    usable_maxy = config.sheet_height_mm - config.sheet_margin_mm

    seen_layer_indices: list[int] = []

    for layout in layouts:
        assert layout.width_mm == config.sheet_width_mm
        assert layout.height_mm == config.sheet_height_mm

        padded_boxes = []
        for placement in layout.placements:
            seen_layer_indices.append(placement.part.layer_index)
            assert placement.sheet_index == layout.sheet_index

            if not config.allow_rotation:
                assert placement.rotation_deg == 0.0
            else:
                assert placement.rotation_deg in (0.0, 90.0)

            minx, miny, maxx, maxy = _materialize_bounds(placement)

            # (a) rotated bbox lies fully within the sheet's usable area.
            assert minx >= usable_minx - EPS
            assert miny >= usable_miny - EPS
            assert maxx <= usable_maxx + EPS
            assert maxy <= usable_maxy + EPS

            half = config.part_spacing_mm / 2.0
            padded_boxes.append((minx - half, miny - half, maxx + half, maxy + half))

        # (b) no two placements on the same sheet have overlapping
        # (padded by part_spacing_mm) bounding boxes.
        for i in range(len(padded_boxes)):
            for j in range(i + 1, len(padded_boxes)):
                assert not _boxes_overlap(padded_boxes[i], padded_boxes[j]), (
                    f"Placements {i} and {j} on sheet {layout.sheet_index} "
                    "overlap once padded by part_spacing_mm"
                )

    # (c) every input part appears exactly once across all returned placements.
    assert sorted(seen_layer_indices) == list(range(len(parts)))


@pytest.mark.parametrize("allow_rotation", [True, False])
def test_nest_parts_forces_multiple_sheets_and_is_valid(allow_rotation: bool) -> None:
    config = Config(
        sheet_width_mm=300.0,
        sheet_height_mm=300.0,
        sheet_margin_mm=5.0,
        part_spacing_mm=3.0,
        allow_rotation=allow_rotation,
    )
    parts = _build_parts(SIZES)

    layouts = nest_parts(parts, config)

    # A 300x300mm sheet is far too small to hold all 24 parts (many of which
    # are individually 80-200mm across) -- this must spill onto multiple sheets.
    assert len(layouts) >= 2

    # Sheet indices are contiguous and match list position.
    assert [layout.sheet_index for layout in layouts] == list(range(len(layouts)))

    _validate_layouts(layouts, parts, config)


def test_nest_parts_prefers_rotation_orientation_with_less_waste() -> None:
    # A shelf established at height 100 by a wide first part; a second,
    # elongated part fits the remaining shelf width only if rotated to be
    # short enough to fit under the established shelf height.
    config = Config(
        sheet_width_mm=300.0,
        sheet_height_mm=150.0,
        sheet_margin_mm=5.0,
        part_spacing_mm=2.0,
        allow_rotation=True,
    )
    parts = [
        Part(layer_index=0, cut_geometry=box(0.0, 0.0, 120.0, 100.0)),
        Part(layer_index=1, cut_geometry=box(0.0, 0.0, 40.0, 90.0)),
    ]
    layouts = nest_parts(parts, config)
    assert len(layouts) == 1
    placements = {p.part.layer_index: p for p in layouts[0].placements}
    # Part 1's natural box is 40x90 (h=90 <= shelf height 100, so it already
    # fits unrotated with waste 10); rotating to 90x40 would waste more (60).
    # So it should be placed unrotated.
    assert placements[1].rotation_deg == 0.0
    _validate_layouts(layouts, parts, config)


def test_nest_parts_simple_two_part_shelf_offsets() -> None:
    config = Config(
        sheet_width_mm=100.0,
        sheet_height_mm=100.0,
        sheet_margin_mm=5.0,
        part_spacing_mm=2.0,
        allow_rotation=False,
    )
    parts = [
        Part(layer_index=0, cut_geometry=box(0.0, 0.0, 30.0, 20.0)),
        Part(layer_index=1, cut_geometry=box(0.0, 0.0, 20.0, 15.0)),
    ]
    layouts = nest_parts(parts, config)
    assert len(layouts) == 1
    assert len(layouts[0].placements) == 2

    by_index = {p.part.layer_index: p for p in layouts[0].placements}
    p0, p1 = by_index[0], by_index[1]

    # Part 0 (larger) is placed first, at the usable area's origin.
    assert p0.rotation_deg == 0.0
    assert p0.x_offset_mm == pytest.approx(5.0)
    assert p0.y_offset_mm == pytest.approx(5.0)

    # Part 1 shares the same shelf (its height 15 <= established shelf height
    # 20), placed to the right of part 0 with exactly part_spacing_mm gap.
    assert p1.rotation_deg == 0.0
    assert p1.x_offset_mm == pytest.approx(5.0 + 30.0 + 2.0)
    assert p1.y_offset_mm == pytest.approx(5.0)

    _validate_layouts(layouts, parts, config)


def test_nest_parts_raises_for_part_too_large_for_any_sheet() -> None:
    config = Config(
        sheet_width_mm=100.0,
        sheet_height_mm=100.0,
        sheet_margin_mm=5.0,
        part_spacing_mm=2.0,
        allow_rotation=True,
    )
    parts = [Part(layer_index=0, cut_geometry=box(0.0, 0.0, 500.0, 500.0))]
    with pytest.raises(ValueError):
        nest_parts(parts, config)
