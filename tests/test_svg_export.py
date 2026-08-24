from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from laser_slice.config import Config
from laser_slice.engraving import layer_number_strokes
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


def test_cut_and_engrave_groups_have_expected_style(tmp_path):
    config = Config()
    sheets = _build_sheets(300.0, 600.0)
    written = export_sheets(sheets, config, str(tmp_path))

    tree = ET.parse(written[0])
    root = tree.getroot()
    layer_group = root.find(f"{SVG_NS}g[@id='layer-0']")
    cut_group = layer_group.find(f"{SVG_NS}g[@id='layer-0-cut']")
    assert cut_group.attrib["stroke"] == "#FF0000"
    assert cut_group.attrib["fill"] == "none"
    assert float(cut_group.attrib["stroke-width"]) == pytest.approx(0.1)

    engrave_group = layer_group.find(f"{SVG_NS}g[@id='layer-0-engrave']")
    assert engrave_group.attrib["stroke"] == "#0000FF"
    assert engrave_group.attrib["fill"] == "none"
    assert float(engrave_group.attrib["stroke-width"]) == pytest.approx(config.engrave_stroke_width_mm)
    assert engrave_group.attrib["stroke-linecap"] == "round"
    assert engrave_group.attrib["stroke-linejoin"] == "round"


def test_each_layer_groups_its_cut_geometry_and_engraving(tmp_path):
    config = Config()
    sheets = _build_sheets(300.0, 600.0)
    written = export_sheets(sheets, config, str(tmp_path))

    root1 = ET.parse(written[0]).getroot()
    layer_groups1 = root1.findall(f"{SVG_NS}g")

    assert [group.attrib["id"] for group in layer_groups1] == ["layer-0", "layer-1"]
    assert len(layer_groups1[0].findall(f"{SVG_NS}g[@id='layer-0-cut']/{SVG_NS}path")) == 1
    assert len(layer_groups1[0].findall(f"{SVG_NS}g[@id='layer-0-engrave']/{SVG_NS}path")) == 2
    assert len(layer_groups1[1].findall(f"{SVG_NS}g[@id='layer-1-cut']/{SVG_NS}path")) == 2
    assert len(layer_groups1[1].findall(f"{SVG_NS}g[@id='layer-1-engrave']/{SVG_NS}path")) == 1

    root2 = ET.parse(written[1]).getroot()
    layer_groups2 = root2.findall(f"{SVG_NS}g")

    assert [group.attrib["id"] for group in layer_groups2] == ["layer-2"]
    assert len(layer_groups2[0].findall(f"{SVG_NS}g[@id='layer-2-cut']/{SVG_NS}path")) == 1
    assert len(layer_groups2[0].findall(f"{SVG_NS}g[@id='layer-2-engrave']/{SVG_NS}path")) == 0

    for root in (root1, root2):
        ids = [element.attrib["id"] for element in root.iter() if "id" in element.attrib]
        assert len(ids) == len(set(ids))


def test_cut_group_path_count_matches_polygon_count(tmp_path):
    config = Config()
    sheets = _build_sheets(300.0, 600.0)
    written = export_sheets(sheets, config, str(tmp_path))

    # Sheet 1: part_a is a single Polygon (with a hole -> still 1 path),
    # part_b is a MultiPolygon of 2 disjoint boxes -> 2 paths. Total = 3.
    tree1 = ET.parse(written[0])
    root1 = tree1.getroot()
    cut_paths1 = [
        path
        for cut_group in root1.findall(f".//{SVG_NS}g[@class='cut']")
        for path in cut_group.findall(f"{SVG_NS}path")
    ]
    assert len(cut_paths1) == 3

    # Each cut path should contain exactly one hole subpath for part_a (2 'M's)
    # and none for the plain boxes (1 'M' each).
    m_counts = sorted(p.attrib["d"].count("M") for p in cut_paths1)
    assert m_counts == [1, 1, 2]

    # Sheet 2: single circular Polygon -> 1 path.
    tree2 = ET.parse(written[1])
    root2 = tree2.getroot()
    cut_paths2 = [
        path
        for cut_group in root2.findall(f".//{SVG_NS}g[@class='cut']")
        for path in cut_group.findall(f"{SVG_NS}path")
    ]
    assert len(cut_paths2) == 1


def test_engrave_group_path_count_matches_glyph_count(tmp_path):
    config = Config()
    sheets = _build_sheets(300.0, 600.0)
    written = export_sheets(sheets, config, str(tmp_path))

    tree1 = ET.parse(written[0])
    root1 = tree1.getroot()
    engrave_paths1 = [
        path
        for engrave_group in root1.findall(f".//{SVG_NS}g[@class='engrave']")
        for path in engrave_group.findall(f"{SVG_NS}path")
    ]
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


def _extract_ordered_points(d: str) -> list[tuple[float, float]]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", d)
    coords = [float(n) for n in nums]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]


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
    cut_group = root.find(f"{SVG_NS}g[@id='layer-0']/{SVG_NS}g[@id='layer-0-cut']")
    paths = cut_group.findall(f"{SVG_NS}path")
    assert len(paths) == 1

    points = _extract_points(paths[0].attrib["d"])
    expected = {(7.0, 10.0), (7.0, 6.0), (5.0, 6.0), (5.0, 10.0)}
    assert points == expected


def test_fitted_engraving_stays_inside_after_svg_rounding(tmp_path):
    config = Config(
        engrave_text_height_mm=8.0,
        engrave_margin_mm=1.0,
        engrave_stroke_width_mm=0.15,
    )
    geometry = box(0.1234564, 0.2345676, 8.1234564, 6.2345676)
    part = Part(
        layer_index=0,
        cut_geometry=geometry,
        engrave_strokes=layer_number_strokes(geometry, layer_index=0, config=config),
    )
    sheet = SheetLayout(
        sheet_index=0,
        width_mm=30.0,
        height_mm=30.0,
        placements=[
            Placement(
                part=part,
                sheet_index=0,
                x_offset_mm=5.1234564,
                y_offset_mm=7.2345676,
                rotation_deg=90.0,
            )
        ],
    )

    [svg_path] = export_sheets([sheet], config, str(tmp_path))
    root = ET.parse(svg_path).getroot()
    cut_path = root.find(
        f"{SVG_NS}g[@id='layer-0']/{SVG_NS}g[@id='layer-0-cut']/{SVG_NS}path"
    )
    engrave_paths = root.findall(
        f"{SVG_NS}g[@id='layer-0']/{SVG_NS}g[@id='layer-0-engrave']/{SVG_NS}path"
    )

    rounded_cut = Polygon(_extract_ordered_points(cut_path.attrib["d"]))
    rounded_engraving = unary_union(
        [LineString(_extract_ordered_points(path.attrib["d"])) for path in engrave_paths]
    )
    stroke_radius = config.engrave_stroke_width_mm / 2.0
    required_centerline_clearance = config.engrave_margin_mm + stroke_radius

    assert rounded_cut.boundary.distance(rounded_engraving) >= (
        required_centerline_clearance - 0.00001
    )
    assert rounded_cut.buffer(0.00001).covers(rounded_engraving.buffer(stroke_radius))
