"""Palette + view-mode color math for the pygame renderer.

frontend/lib/replay/shared.ts is the TypeScript mirror of every constant and formula below
(same hash constants, same HSL formulas, same measurement ramp) - so a brick's jitter and an
audit's tint are bit-for-bit the same computation whether it's drawn by pygame, the 2D
canvas, or three.js. See that file's own module comment for the four-system rationale
(SUBSTRATE/MATERIAL/the audit's ramp/INTENT); pygame has no page chrome, so this file only
carries MATERIAL, INTENT, and the ramp.

Values are plain (r, g, b) 0-255 int tuples throughout, since that's what pygame consumes
directly - the frontend's copy stores hex strings and 0..1 floats instead, per its own
renderers' needs.
"""

from __future__ import annotations

from atrium_sim.constants import TOL_MM


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# --- MATERIAL: what things ARE, independent of measurement -------------------------------
BG = _hex("#0F0E0D")
GROUND = _hex("#3C3733")
MORTAR = _hex("#7A7269")
CLAY = _hex("#B4593C")
CLAY_FALLEN = _hex("#4A2B21")  # a toppled brick - unlit clay in shadow, not an error code
STONE = _hex("#B8B0A2")  # skewback / abutment wedges, lintels, sills
STONE_EDGE = _hex("#83796C")
TIMBER = _hex("#7E6136")  # temporary arch centering - visible only until struck
TIMBER_EDGE = _hex("#55401F")
CEMENT = _hex("#8F8B85")  # cement lintel/sill heads
ROBOT = _hex("#7E858E")  # the gantry body - equipment, not a toy
ROBOT_DARK = _hex("#565C63")
ROBOT_TOOL = _hex("#FFB020")  # the ONE thing amber still means: the active tool/gripper

# --- INTENT: the drafting/ghost layer -----------------------------------------------------
CHALK = _hex("#8E9AA8")  # the only cool hue in as-built/drawing mode
GHOST_OPACITY = 0.22
NEXT_SLOT_OPACITY = 0.5

# --- UI (HUD text, never a brick) ---------------------------------------------------------
ACCENT = _hex("#F2B94B")
HUD_BG = _hex("#0C0B0A")
HUD_TEXT = _hex("#EDE9E3")
LABEL = _hex("#9A938A")


def blend(fg: tuple[int, int, int], bg: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Flatten a translucent color onto a background. pygame's draw.polygon/draw.line have no
    per-primitive alpha, so ghost/next-slot targets are pre-blended to a solid RGB here
    instead of drawn with a real alpha channel (which the browser renderers use directly)."""
    return tuple(round(bg[i] + (fg[i] - bg[i]) * alpha) for i in range(3))  # type: ignore[return-value]


# --- per-brick clay jitter ----------------------------------------------------------------
# triple32, a public-domain 32-bit integer bit-mixer - see shared.ts's hash01 for why this
# (not a PRNG library) is what guarantees the two languages compute the identical value from
# the same brick_id.
_MASK32 = 0xFFFFFFFF


def _hash01(n: int) -> float:
    x = (n ^ 0x9E3779B9) & _MASK32
    x = ((x ^ (x >> 16)) * 0x045D9F3B) & _MASK32
    x = ((x ^ (x >> 16)) * 0x045D9F3B) & _MASK32
    x = (x ^ (x >> 16)) & _MASK32
    return x / 4294967296.0


def _rgb_to_hsl(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    hi, lo = max(r, g, b), min(r, g, b)
    l = (hi + lo) / 2
    if hi == lo:
        return (0.0, 0.0, l)
    d = hi - lo
    s = d / (2 - hi - lo) if l > 0.5 else d / (hi + lo)
    if hi == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif hi == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return (h / 6, s, l)


def _hue_to_rgb_channel(p: float, q: float, t: float) -> float:
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1 / 6:
        return p + (q - p) * 6 * t
    if t < 1 / 2:
        return q
    if t < 2 / 3:
        return p + (q - p) * (2 / 3 - t) * 6
    return p


def _hsl_to_rgb(hsl: tuple[float, float, float]) -> tuple[float, float, float]:
    h, s, l = hsl
    if s == 0:
        return (l, l, l)
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (
        _hue_to_rgb_channel(p, q, h + 1 / 3),
        _hue_to_rgb_channel(p, q, h),
        _hue_to_rgb_channel(p, q, h - 1 / 3),
    )


def clay_jitter(brick_id: int, base: tuple[int, int, int] = CLAY) -> tuple[int, int, int]:
    """Deterministic per-brick clay variation, computed from brick_id alone - a wall of
    identical boxes reads as plastic; +-4deg hue / +-8% saturation / +-7% lightness is enough
    to read as fired clay without any single brick looking like a mistake. See shared.ts's
    clayJitterRgb for the identical TypeScript formula."""
    dh = (_hash01(brick_id * 3 + 0) - 0.5) * 2 * (4 / 360)
    ds = (_hash01(brick_id * 3 + 1) - 0.5) * 2 * 0.08
    dl = (_hash01(brick_id * 3 + 2) - 0.5) * 2 * 0.07
    h, s, l = _rgb_to_hsl(tuple(c / 255 for c in base))
    r, g, b = _hsl_to_rgb((((h + dh) % 1 + 1) % 1, min(1.0, max(0.0, s + ds)), min(1.0, max(0.0, l + dl))))
    return (round(r * 255), round(g * 255), round(b * 255))


# --- the audit's ramp (inspect mode only) --------------------------------------------------
# Signed and diverging, per the scan-vs-BIM deviation-heatmap convention. dx = brick.x -
# target.x is the brick's own signed lateral offset, not a material-excess/deficit read the
# way a surface-scan deviation is - a brick built one way along the wall isn't "worse" than
# the other, so the teal/amber split just needs to be CONSISTENT, not physically meaningful.
# Within tolerance is deliberately low-contrast: a correct wall should look calm, and only
# the outliers should pull the eye.
_MEASURE_NEUTRAL = _hex("#8A857D")  # 0mm - a desaturated clay-grey, not paper white
_MEASURE_TEAL_SOFT = _hex("#6FA9AC")  # right at -TOL_MM
_MEASURE_TEAL_STRONG = _hex("#2F8990")  # far under
_MEASURE_AMBER_SOFT = _hex("#B99361")  # right at +TOL_MM
_MEASURE_RED_STRONG = _hex("#C24A3F")  # far over
MEASURE_FALLOFF_MM = 15.0  # distance past TOL_MM over which soft ramps to strong


def _lerp3(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def measurement_color(dx: float, in_tol: bool) -> tuple[int, int, int]:
    ad = abs(dx)
    soft = _MEASURE_AMBER_SOFT if dx >= 0 else _MEASURE_TEAL_SOFT
    if in_tol:
        return _lerp3(_MEASURE_NEUTRAL, soft, (ad / TOL_MM) * 0.6)
    strong = _MEASURE_RED_STRONG if dx >= 0 else _MEASURE_TEAL_STRONG
    return _lerp3(soft, strong, (ad - TOL_MM) / MEASURE_FALLOFF_MM)


def brick_color(
    mode: str, brick_id: int, status: str, dx: float | None, in_tol: bool | None,
) -> tuple[int, int, int]:
    """mode: 'as-built' | 'inspect' | 'drawing'; status: 'matched' | 'flight' | 'stray'.
    Mirrors shared.ts's brickColorRgb exactly - see that function's docstring for the rule
    table. Pygame's 'as-built' can only be a flat fill (no lighting, no environment map),
    which is expected: the GIFs are the one place this identity trades material realism for
    parity with the browser renderers."""
    if status == "stray":
        return clay_jitter(brick_id, CLAY_FALLEN)
    if mode == "drawing":
        return CHALK
    if mode == "inspect":
        if status == "matched" and dx is not None:
            return measurement_color(dx, bool(in_tol))
        return _MEASURE_NEUTRAL
    return clay_jitter(brick_id, CLAY)  # as-built
