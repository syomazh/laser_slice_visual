"""Nest a list of Parts onto one or more sheets using First-Fit-Decreasing-Height
(FFDH) shelf bin-packing.

See laser_slice/geometry_types.py for the exact Placement coordinate contract
(rotate-about-bbox-center, then translate so the rotated bbox's min corner
lands at (x_offset_mm, y_offset_mm), sheet origin at the bottom-left corner).
This module must follow that contract exactly, since svg_export.py (built
independently) relies on it verbatim.
"""
from __future__ import annotations

from laser_slice.config import Config
from laser_slice.geometry_types import Part, Placement, SheetLayout

# Tolerance for floating point comparisons when deciding whether a box fits.
_EPS = 1e-6


class _Shelf:
    __slots__ = ("y_bottom", "height", "x_cursor")

    def __init__(self, y_bottom: float, height: float) -> None:
        self.y_bottom = y_bottom
        self.height = height
        self.x_cursor = 0.0  # next free x position, usable-area-relative


class _Sheet:
    __slots__ = ("shelves", "placements")

    def __init__(self) -> None:
        self.shelves: list[_Shelf] = []
        self.placements: list[Placement] = []


def _bbox_size(part: Part) -> tuple[float, float]:
    minx, miny, maxx, maxy = part.cut_geometry.bounds
    return maxx - minx, maxy - miny


def _candidate_orientations(
    w: float, h: float, allow_rotation: bool
) -> list[tuple[float, float, float]]:
    """Return (box_w, box_h, rotation_deg) candidates for a part's bounding box."""
    candidates = [(w, h, 0.0)]
    if allow_rotation:
        candidates.append((h, w, 90.0))
    return candidates


def _shelf_needed_width(shelf: _Shelf, box_w: float, spacing: float) -> float:
    """Width shelf.x_cursor must advance by to place box_w (including any
    inter-part spacing needed before it)."""
    if shelf.x_cursor <= _EPS:
        return box_w
    return spacing + box_w


def nest_parts(parts: list[Part], config: Config) -> list[SheetLayout]:
    """Pack `parts` onto sheets of size config.sheet_width_mm x
    config.sheet_height_mm using First-Fit-Decreasing-Height shelf packing.

    Returns one SheetLayout per sheet actually used, each containing the
    Placements for the parts nested onto it, in the order they were placed.
    """
    margin = config.sheet_margin_mm
    spacing = config.part_spacing_mm
    usable_w = config.sheet_width_mm - 2.0 * margin
    usable_h = config.sheet_height_mm - 2.0 * margin

    if usable_w <= 0 or usable_h <= 0:
        raise ValueError(
            "Sheet usable area is empty after applying sheet_margin_mm; "
            f"sheet={config.sheet_width_mm}x{config.sheet_height_mm}, margin={margin}"
        )

    # Sort parts by decreasing max(w, h) (decreasing "size"). Rotation swaps
    # w/h, so max(w, h) is rotation-invariant and safe to sort on directly.
    sized_parts = [(part, *_bbox_size(part)) for part in parts]
    sized_parts.sort(key=lambda item: max(item[1], item[2]), reverse=True)

    sheets: list[_Sheet] = []

    for part, w, h in sized_parts:
        candidates = _candidate_orientations(w, h, config.allow_rotation)

        # A part that cannot fit on a completely empty sheet (in any allowed
        # orientation) can never be nested at all.
        if not any(
            cw <= usable_w + _EPS and ch <= usable_h + _EPS
            for cw, ch, _ in candidates
        ):
            raise ValueError(
                f"Part (layer_index={part.layer_index}) with bounding box "
                f"{w:.3f}x{h:.3f}mm does not fit on an empty sheet "
                f"({usable_w:.3f}x{usable_h:.3f}mm usable area)."
            )

        placed = False

        # 1. Try existing shelves, first-fit, across all sheets in creation order.
        for sheet_idx, sheet in enumerate(sheets):
            for shelf in sheet.shelves:
                remaining_width = usable_w - shelf.x_cursor
                fitting = []
                for cw, ch, rot in candidates:
                    needed_width = _shelf_needed_width(shelf, cw, spacing)
                    if needed_width <= remaining_width + _EPS and ch <= shelf.height + _EPS:
                        waste = shelf.height - ch
                        fitting.append((waste, needed_width, cw, ch, rot))
                if fitting:
                    # Prefer the orientation with the least wasted shelf height.
                    fitting.sort(key=lambda item: item[0])
                    _waste, needed_width, cw, ch, rot = fitting[0]
                    x_local = shelf.x_cursor + (needed_width - cw)
                    y_local = shelf.y_bottom
                    shelf.x_cursor = x_local + cw
                    sheet.placements.append(
                        Placement(
                            part=part,
                            sheet_index=sheet_idx,
                            x_offset_mm=margin + x_local,
                            y_offset_mm=margin + y_local,
                            rotation_deg=rot,
                        )
                    )
                    placed = True
                    break
            if placed:
                break

        # 2. Try opening a new shelf below the last one, on the current
        # (most recently created) sheet, if vertical room remains.
        if not placed and sheets:
            sheet = sheets[-1]
            if sheet.shelves:
                last_shelf = sheet.shelves[-1]
                next_y = last_shelf.y_bottom + last_shelf.height + spacing
            else:
                next_y = 0.0
            avail_height = usable_h - next_y

            fitting = []
            for cw, ch, rot in candidates:
                if cw <= usable_w + _EPS and ch <= avail_height + _EPS:
                    fitting.append((ch, cw, rot))
            if fitting:
                # Prefer the orientation using the least vertical height
                # (minimizes waste of remaining sheet height for future shelves).
                fitting.sort(key=lambda item: item[0])
                ch, cw, rot = fitting[0]
                new_shelf = _Shelf(y_bottom=next_y, height=ch)
                new_shelf.x_cursor = cw
                sheet.shelves.append(new_shelf)
                sheet.placements.append(
                    Placement(
                        part=part,
                        sheet_index=len(sheets) - 1,
                        x_offset_mm=margin + 0.0,
                        y_offset_mm=margin + next_y,
                        rotation_deg=rot,
                    )
                )
                placed = True

        # 3. Open a new sheet.
        if not placed:
            fitting = []
            for cw, ch, rot in candidates:
                if cw <= usable_w + _EPS and ch <= usable_h + _EPS:
                    fitting.append((ch, cw, rot))
            # Guaranteed non-empty by the up-front empty-sheet-fit check above.
            fitting.sort(key=lambda item: item[0])
            ch, cw, rot = fitting[0]

            new_sheet = _Sheet()
            new_shelf = _Shelf(y_bottom=0.0, height=ch)
            new_shelf.x_cursor = cw
            new_sheet.shelves.append(new_shelf)
            sheet_index = len(sheets)
            new_sheet.placements.append(
                Placement(
                    part=part,
                    sheet_index=sheet_index,
                    x_offset_mm=margin + 0.0,
                    y_offset_mm=margin + 0.0,
                    rotation_deg=rot,
                )
            )
            sheets.append(new_sheet)
            placed = True

        assert placed  # pragma: no cover - defensive; unreachable by construction

    return [
        SheetLayout(
            sheet_index=idx,
            width_mm=config.sheet_width_mm,
            height_mm=config.sheet_height_mm,
            placements=sheet.placements,
        )
        for idx, sheet in enumerate(sheets)
    ]
