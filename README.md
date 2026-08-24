# laser_slice

Turn a 3D model into a stack of horizontal, laser-cuttable layers: load an OBJ/STL,
slice it into slabs matching your material thickness, auto-nest the layer
silhouettes onto sheets, optionally punch dowel/registration holes so the stack
lines up when glued, optionally engrave each layer with its layer number, and
export ready-to-cut SVGs.

Only the "stacked horizontal" construction technique is implemented (each
layer is one flat slab cut from sheet stock and glued on top of the next).

## Install

```
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e .
```

This pulls in `trimesh`, `manifold3d` (mesh booleans, no Blender needed),
`shapely`, `svgwrite`, `numpy`, `matplotlib`, `scipy`, `networkx`, `rtree`.

## Quick start

Try it on the bundled sample figurine (a torso/head/arms/legs test shape --
regenerate it any time with `./.venv/Scripts/python.exe examples/make_sample.py`):

```
./.venv/Scripts/python.exe -m laser_slice.cli examples/sample.obj --out output
```

This slices at the default 3mm (basswood) thickness, nests onto a 300x600mm
sheet, cuts registration holes, engraves layer numbers, writes
`output/sheet_01.svg` (and more if it doesn't fit on one sheet) plus
`output/overview.png`, and finally opens an interactive matplotlib viewer
(close the window to let the process exit; pass `--no-visualize` to skip it
for headless/batch runs).

## CLI options

```
laser-slice INPUT.obj
  --thickness MM        material thickness (default 3.0, e.g. 3mm basswood)
  --sheet WxH            sheet size in mm, e.g. 300x600 (default 300x600)
  --margin MM            keep-out margin from sheet edges (default 5.0)
  --spacing MM           minimum gap between nested parts (default 3.0)
  --kerf MM              total laser kerf width to compensate for (default 0.0)
  --axis {x,y,z}         model axis to stack layers along (default z)
  --height-mm H          uniformly scale the model so its extent along --axis is H mm
  --scale S              uniformly scale raw model coordinates by S (alternative to --height-mm)
  --no-rotation          disable 90-degree part rotation while nesting
  --no-registration      disable registration (dowel) holes
  --reg-diameter MM      registration hole diameter (default 3.0, e.g. for a 1/8" dowel use 3.175)
  --reg-margin MM        required clearance around a registration hole (default 4.0)
  --no-engrave           disable engraving of layer numbers
  --engrave-height MM    engraved digit height (default 4.0)
  --engrave-margin MM    distance from a part's edge to its engraved number (default 3.0)
  --out DIR              output directory (default output/)
  --no-visualize         skip the interactive viewer at the end
  --no-stack-preview     skip the physically-reconstructed stack preview (PNG + OBJ)
```

Also runnable as `python -m laser_slice ...` once installed.

### Sizing your model

OBJ/STL files carry no notion of real-world units. Use exactly one of
`--height-mm` (scale so the model's height along the stacking axis is a
specific size) or `--scale` (multiply raw coordinates directly); if you omit
both, coordinates are assumed to already be millimeters.

## Output / SVG conventions (LightBurn-ready)

Each sheet is one SVG sized in real millimeters (`viewBox` matches
width/height 1:1, so it imports at true physical scale). Two layers:

- **Cut** (`id="cut"`, red `#FF0000` stroke, no fill) -- the outer silhouette
  of every part plus every hole (object holes from the mesh cross-section,
  and registration holes), each part as one `<path>` with the exterior ring
  and every hole ring as separate subpaths.
- **Engrave** (`id="engrave"`, blue `#0000FF` stroke) -- the vector-stroke
  layer number for each part, positioned near its bottom-left corner.

In LightBurn, map the red layer to a Cut operation and the blue layer to a
Line/engrave operation.

### Kerf compensation

`--kerf` is the *total* width your laser removes (e.g. measure a test cut).
Leave it at 0 to cut at nominal size. Internally the whole cut geometry
(outer boundary + every hole) is dilated by `kerf/2` in one operation, which
simultaneously pushes outlines outward and shrinks holes inward by the right
amount so the finished, physically-cut part matches your nominal design size.

### Registration holes

By default, two dowel holes are auto-placed using the layer with the largest
cross-sectional area (typically a torso/trunk/base -- present in the most
layers) rather than the bounding box of the whole model, so a model with thin
limbs sticking out doesn't drag the holes into empty space. Each layer
independently only gets a hole cut where it actually has enough surrounding
material (radius + `--reg-margin`); the CLI's summary at the end lists which
layers had a hole skipped. If the auto-placement doesn't suit your model,
set `Config(registration_points=[(x, y), ...])` (in your own script, or by
importing and calling `laser_slice.cli.run` with a custom `Config`) to pin
exact dowel locations in the model's centered XY frame.

## Visualizer

`output/overview.png` is always written (a static 3D view of the mesh with
layer-boundary planes, plus a thumbnail grid of every layer's silhouette --
useful for a quick headless sanity check). Unless `--no-visualize` is passed,
an interactive matplotlib window also opens with a slider to scrub through
layers, showing the 3D slice plane, that layer's 2D shape + engraving, and
its position on its nested sheet.

### "What will it actually look like?" -- the stack preview

The original mesh is smooth; the physical object won't be -- it's built from
flat horizontal slabs, so it has a visible stairstep on any sloped/curved
surface. Unless `--no-stack-preview` is passed, laser_slice reconstructs the
*actual* physical geometry by extruding every layer's final, already-cut
(kerf-compensated, holes-and-all) shape to its real slab thickness and
stacking the slabs at their true height (adjacent layers alternate two wood
tones so individual layers are visible, like real glued veneer):

- `output/stack_preview.png` -- a static render, always written.
- `output/stack_preview.obj` -- the actual reconstructed 3D mesh; open it in
  Blender, MeshLab, or any 3D viewer to freely rotate/zoom and see exactly
  what the glued-up sculpture will look like before you cut anything.
- An interactive matplotlib window (rotate with the mouse) also opens
  alongside the layer-scrubbing viewer, unless `--no-visualize` is passed.

## Project layout

```
laser_slice/
  config.py          Config dataclass (all tunables, defaults to 3mm basswood)
  geometry_types.py  shared data types + the nesting/SVG coordinate contract
  mesh_io.py         load + canonicalize (axis, scale, centering) a mesh
  slicer.py          horizontal cross-sections -> per-layer shapely geometry
  polygon_ops.py     cleanup + kerf compensation
  registration.py    dowel-hole point selection + cutting
  nesting.py         bounding-box shelf packing onto sheets
  fonts.py           hand-coded vector digit font for engraving
  svg_export.py      final SVG writer
  stack_preview.py   reconstructs the actual physical (stairstepped) 3D stack
  visualizer.py      matplotlib overview + interactive viewer + stack preview
  cli.py             wires the whole pipeline together
tests/               pytest suite (47 tests) covering every module above
examples/            make_sample.py generates the bundled test figurine
```

Run the test suite with `./.venv/Scripts/python.exe -m pytest tests/ -v`.

## Known limitations (v1)

- Only stacked-horizontal is implemented (no contour/other construction modes).
- Nesting is bounding-box based, not true irregular/contour nesting -- simple
  and reliable, but leaves some empty sheet space around non-rectangular parts.
- The registration-hole heuristic anchors to the single largest-area layer;
  for unusual shapes with no clear "core" (e.g. a tall thin vase that tapers
  continuously), consider setting `registration_points` explicitly.
