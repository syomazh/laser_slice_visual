"""A tiny hand-coded single-stroke ("vector plotter" / Hershey-style) font.

Only digits 0-9 are defined (this module exists to engrave layer numbers),
plus a placeholder box for any other character so callers never need to
special-case unsupported input.

Each glyph is defined on a unit grid roughly 0.6 units wide by 1.0 units
tall, as one or more open polylines ("strokes"). ``text_to_strokes`` lays
characters out left-to-right starting at ``origin``, scales the unit grid so
the glyph height equals ``height_mm``, and returns a flat list of
``EngraveGlyph`` with coordinates already in final position.
"""
from __future__ import annotations

from laser_slice.geometry_types import EngraveGlyph

# --- unit grid coordinates -------------------------------------------------
# x positions (glyph is ~0.6 units wide)
X_L = 0.0
X_LM = 0.15
X_M = 0.3
X_RM = 0.45
X_R = 0.6

# y positions (glyph is 1.0 units tall)
Y_B = 0.0
Y_BQ = 0.25
Y_MID = 0.5
Y_TQ = 0.75
Y_T = 1.0

Point = tuple[float, float]
Stroke = list[Point]

# Each digit maps to a list of strokes (open polylines in unit-grid space).
_DIGIT_STROKES: dict[str, list[Stroke]] = {
    "0": [
        [
            (X_LM, Y_B), (X_RM, Y_B), (X_R, Y_BQ), (X_R, Y_TQ),
            (X_RM, Y_T), (X_LM, Y_T), (X_L, Y_TQ), (X_L, Y_BQ), (X_LM, Y_B),
        ],
    ],
    "1": [
        [(X_M, Y_B), (X_M, Y_T)],
        [(X_LM, Y_TQ), (X_M, Y_T)],
        [(X_LM, Y_B), (X_RM, Y_B)],
    ],
    "2": [
        [
            (X_L, Y_TQ), (X_LM, Y_T), (X_RM, Y_T), (X_R, Y_TQ),
            (X_R, Y_MID), (X_L, Y_B), (X_R, Y_B),
        ],
    ],
    "3": [
        [
            (X_L, Y_T), (X_R, Y_T), (X_R, Y_MID), (X_M, Y_MID),
            (X_R, Y_MID), (X_R, Y_B), (X_L, Y_B),
        ],
    ],
    "4": [
        [(X_RM, Y_T), (X_L, Y_BQ), (X_R, Y_BQ)],
        [(X_RM, Y_T), (X_RM, Y_B)],
    ],
    "5": [
        [
            (X_R, Y_T), (X_L, Y_T), (X_L, Y_MID), (X_RM, Y_MID),
            (X_R, Y_BQ), (X_RM, Y_B), (X_LM, Y_B),
        ],
    ],
    "6": [
        [
            (X_RM, Y_T), (X_LM, Y_TQ), (X_L, Y_MID), (X_L, Y_BQ),
            (X_LM, Y_B), (X_RM, Y_B), (X_R, Y_BQ), (X_RM, Y_MID), (X_LM, Y_MID),
        ],
    ],
    "7": [
        [(X_L, Y_T), (X_R, Y_T)],
        [(X_R, Y_T), (X_LM, Y_B)],
    ],
    "8": [
        [
            (X_LM, Y_MID), (X_RM, Y_MID), (X_R, Y_TQ), (X_RM, Y_T),
            (X_LM, Y_T), (X_L, Y_TQ), (X_LM, Y_MID),
        ],
        [
            (X_LM, Y_MID), (X_RM, Y_MID), (X_R, Y_BQ), (X_RM, Y_B),
            (X_LM, Y_B), (X_L, Y_BQ), (X_LM, Y_MID),
        ],
    ],
    "9": [
        [
            (X_RM, Y_MID), (X_LM, Y_MID), (X_L, Y_TQ), (X_LM, Y_T),
            (X_RM, Y_T), (X_R, Y_TQ), (X_R, Y_MID), (X_RM, Y_B), (X_LM, Y_BQ),
        ],
    ],
}

# Simple placeholder box for anything that isn't a digit (e.g. letters,
# punctuation, whitespace) -- rendered rather than raising.
_PLACEHOLDER_STROKES: list[Stroke] = [
    [(X_L, Y_B), (X_R, Y_B), (X_R, Y_T), (X_L, Y_T), (X_L, Y_B)],
]

GLYPH_WIDTH_UNITS = 0.6
GLYPH_GAP_UNITS = 0.2
ADVANCE_UNITS = GLYPH_WIDTH_UNITS + GLYPH_GAP_UNITS


def text_to_strokes(
    text: str,
    height_mm: float,
    origin: tuple[float, float] = (0.0, 0.0),
) -> list[EngraveGlyph]:
    """Convert ``text`` into a flat list of scaled/positioned EngraveGlyph strokes.

    Characters are laid out left-to-right starting at ``origin``, each
    scaled so the glyph height equals ``height_mm``, with a fixed advance
    width (glyph width + gap) between characters. Characters outside
    '0'-'9' are rendered as a simple placeholder box.
    """
    glyphs: list[EngraveGlyph] = []
    advance_mm = ADVANCE_UNITS * height_mm
    ox, oy = origin

    for i, ch in enumerate(text):
        strokes = _DIGIT_STROKES.get(ch, _PLACEHOLDER_STROKES)
        cursor_x = ox + i * advance_mm
        for stroke in strokes:
            points = [(cursor_x + ux * height_mm, oy + uy * height_mm) for ux, uy in stroke]
            glyphs.append(EngraveGlyph(points=points))

    return glyphs
