from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from laser_slice.config import Config
from laser_slice.geometry_types import EngraveGlyph, Part, Placement, SheetLayout
from laser_slice.svg_export import export_sheets

SVG_NS = "{http://www.w3.org/2000/svg}"


def _make_hole_polygon() -> Polygon:
    # 10x10 square with a 2x2 square hole in the middle: one Polygon, one interior ring.
    shell = box(0, 0, 10, 10).exterior.coords
    hole = box(4, 4, 6, 6).exterior.coords
    return Polygon(shell, [hole])


def _make_multipolygon() -> MultiPolygon:
    return MultiPolygon([box(0, 0, 3, 3), box(20, 20, 25, 25)])


def _build_sheets(width_mm: float, height_mm: float):
    part_a = Part(
        layer_index=0,
        cut_geometry=_make_hole_polygon(),
        engrave_strokes=[
            EngraveGlyph(points=[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]),
            EngraveGlyph(points=[(3.0, 1.0), (3.0, 2.0)]),
        ],
    )
    part_b = Part(
        layer_index=1,
        cut_geometry=_make_multipolygon(),
        engrave_strokes=[EngraveGlyph(points=[(0.5, 0.5), (1.5, 0.5)])],
    )

    sheet1 = SheetLayout(
        sheet_index=0,
        width_mm=width_mm,
        height_mm=height_mm,
        placements=[
            Placement(part=part_a, sheet_index=0, x_offset_mm=10.0, y_offset_mm=10.0, rotation_deg=0.0),
            Placement(part=part_b, sheet_index=0, x_offset_mm=50.0, y_offset_mm=50.0, rotation_deg=0.0),
        ],
    )

    part_c = Part(
        layer_index=2,
        cut_geometry=Point(0, 0).buffer(4.0, quad_segs=8),
        engrave_strokes=[],
    )
    sheet2 = SheetLayout(
        sheet_index=1,
        width_mm=width_mm,
        height_mm=height_mm,
        placements=[
            Placement(part=part_c, sheet_index=1, x_offset_mm=5.0, y_offset_mm=5.0, rotation_deg=90.0),
        ],
    )

    return [sheet1, sheet2]


def test_export_sheets_creates_out_dir_and_files(tmp_path):
    config = Config()
    out_dir = os.path.join(str(tmp_path), "nested", "out")
    sheets = _build_sheets(300.0, 600.0)

    written = export_sheets(sheets, config, out_dir)

    assert os.path.isdir(out_dir)
    assert len(written) == 2
    for path in written:
        assert os.path.isfile(path)


def test_svg_well_formed_and_root_matches_sheet_size(tmp_path):
    config = Config()
    width_mm, height_mm = 300.0, 600.0
    sheets = _build_sheets(width_mm, height_mm)
    written = export_sheets(sheets, config, str(tmp_path))

    for path, sheet in zip(written, sheets):
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == f"{SVG_NS}svg"
        assert root.attrib["width"] == f"{sheet.width_mm}mm"
        assert root.attrib["height"] == f"{sheet.height_mm}mm"
        assert root.attrib["viewBox"] == f"0 0 {sheet.width_mm} {sheet.height_mm}"


def test_cut_and_engrave_groups_exist_with_expected_style(tmp_path):
    config = Config()
    sheets = _build_sheets(300.0, 600.0)
    written = export_sheets(sheets, config, str(tmp_path))

    tree = ET.parse(written[0])
    root = tree.getroot()
    groups = {g.attrib.get("id"): g for g in root.findall(f"{SVG_NS}g")}

    assert "cut" in groups
    assert "engrave" in groups

    cut_group = groups["cut"]
    assert cut_group.attrib["stroke"] == "#FF0000"
    assert cut_group.attrib["fill"] == "none"
    assert float(cut_group.attrib["stroke-width"]) == pytest.approx(0.1)

    engrave_group = groups["engrave"]
    assert engrave_group.attrib["stroke"] == "#0000FF"
    assert engrave_group.attrib["fill"] == "none"
    assert float(engrave_group.attrib["stroke-width"]) == pytest.approx(config.engrave_stroke_width_mm)


def test_cut_group_path_count_matches_polygon_count(tmp_path):
    config = Config()
    sheets = _build_sheets(300.0, 600.0)
    written = export_sheets(sheets, config, str(tmp_path))

    # Sheet 1: part_a is a single Polygon (with a hole -> still 1 path),
    # part_b is a MultiPolygon of 2 disjoint boxes -> 2 paths. Total = 3.
    tree1 = ET.parse(written[0])
    root1 = tree1.getroot()
    cut_group1 = root1.find(f"{SVG_NS}g[@id='cut']")
    cut_paths1 = cut_group1.findall(f"{SVG_NS}path")
    assert len(cut_paths1) == 3

    # Each cut path should contain exactly one hole subpath for part_a (2 'M's)
    # and none for the plain boxes (1 'M' each).
    m_counts = sorted(p.attrib["d"].count("M") for p in cut_paths1)
    assert m_counts == [1, 1, 2]

    # Sheet 2: single circular Polygon -> 1 path.
    tree2 = ET.parse(written[1])
    root2 = tree2.getroot()
    cut_group2 = root2.find(f"{SVG_NS}g[@id='cut']")
    cut_paths2 = cut_group2.findall(f"{SVG_NS}path")
    assert len(cut_paths2) == 1


def test_engrave_group_path_count_matches_glyph_count(tmp_path):
    config = Config()
    sheets = _build_sheets(300.0, 600.0)
    written = export_sheets(sheets, config, str(tmp_path))

    tree1 = ET.parse(written[0])
    root1 = tree1.getroot()
    engrave_group1 = root1.find(f"{SVG_NS}g[@id='engrave']")
    engrave_paths1 = engrave_group1.findall(f"{SVG_NS}path")
    # part_a has 2 glyphs, part_b has 1 glyph => 3 total on sheet 1.
    assert len(engrave_paths1) == 3
    for p in engrave_paths1:
        d = p.attrib["d"]
        assert d.startswith("M")
        assert "Z" not in d


def _extract_points(d: str) -> set[tuple[float, float]]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", d)
    coords = [float(n) for n in nums]
    return {(round(coords[i], 3), round(coords[i + 1], 3)) for i in range(0, len(coords), 2)}


def test_placement_rotate_translate_and_y_flip_contract(tmp_path):
    # A 4x2 box rotated 90 degrees about its own bbox center, then translated
    # so the rotated bbox's min corner sits at (5, 10), on a sheet of height 20.
    # Worked out by hand per the Placement docstring recipe (see PR notes).
    config = Config()
    part = Part(layer_index=0, cut_geometry=box(0, 0, 4, 2), engrave_strokes=[])
    sheet = SheetLayout(
        sheet_index=0,
        width_mm=50.0,
        height_mm=20.0,
        placements=[
            Placement(part=part, sheet_index=0, x_offset_mm=5.0, y_offset_mm=10.0, rotation_deg=90.0),
        ],
    )
    written = export_sheets([sheet], config, str(tmp_path))

    tree = ET.parse(written[0])
    root = tree.getroot()
    cut_group = root.find(f"{SVG_NS}g[@id='cut']")
    paths = cut_group.findall(f"{SVG_NS}path")
    assert len(paths) == 1

    points = _extract_points(paths[0].attrib["d"])
    expected = {(7.0, 10.0), (7.0, 6.0), (5.0, 6.0), (5.0, 10.0)}
    assert points == expected
