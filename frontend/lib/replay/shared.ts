// Palette + pure geometry/color helpers shared by BOTH replay renderers (draw.ts's 2D
// canvas port and scene3d/SceneCanvas's 3D scene) - one source of truth for "what does
// atrium_sim/render/palette.py's palette and the view-mode color rules actually say", so the
// two renderers can never silently drift into looking like different products.
//
// The palette is deliberately split into four systems that used to share hue and therefore
// fought each other (terracotta-the-clay vs green-the-status vs amber-the-everything):
//   SUBSTRATE   - the site itself: bg/panel/ink, never seen on a brick.
//   MATERIAL    - what things ARE (clay, mortar, stone, timber, steel) - what the wall looks
//                 like when nothing is being measured.
//   the audit's ramp (see measurementRgb) - what the audit SAYS, signed and diverging,
//                 visible ONLY in "inspect" mode. This is the layer BIM deviation reports use
//                 (teal/paper/amber-red), and it used to be the ONLY layer this project had -
//                 every brick painted pass/fail green, all the time.
//   INTENT      - the drafting/ghost layer (unbuilt targets): one hue, chalk, so it always
//                 reads as "not yet real" rather than competing with either of the above.
// ACCENT is UI brand/focus only - it never appears on a brick.

export type ViewMode = "as-built" | "inspect" | "drawing";
export const VIEW_MODES: readonly ViewMode[] = ["as-built", "inspect", "drawing"];

export const SUBSTRATE = {
  bg: "#0F0E0D",
  panel: "#1C1A18",
  panel2: "#242220",
  line: "#332F2B",
  lineStrong: "#4A443E",
  ink: "#EDE9E3",
  muted: "#9A938A",
  faint: "#6B655D",
} as const;

export const MATERIAL = {
  clay: "#B4593C",
  clayDark: "#8E432C",
  clayLight: "#C97452",
  clayFallen: "#4A2B21", // a toppled brick - unlit clay in shadow, not an error code
  mortar: "#7A7269",
  stone: "#B8B0A2", // skewback / abutment wedges, lintels, sills
  stoneEdge: "#83796C",
  timber: "#7E6136", // temporary arch centering - visible only until struck
  timberEdge: "#55401F",
  concrete: "#8F8B85", // cement lintel/sill heads
  ground: "#3C3733",
  steel: "#7E858E", // the gantry body - equipment, not a toy
  steelDark: "#565C63",
  machineAmber: "#FFB020", // the ONE thing amber still means: the active tool/gripper
} as const;

export const INTENT = {
  chalk: "#8E9AA8", // the only cool hue in as-built/drawing mode - unbuilt reads as "not real"
} as const;

export const ACCENT = "#F2B94B"; // brand, focus rings, active toggles - never on a brick

/** Ghost target opacity: unfilled targets recede to context instead of dominating the frame
 * (was a flat 0.55 for every target, which is why old stills were littered with 600 dashed
 * boxes). The next expected slot stays notably brighter than the rest of the blueprint. */
export const GHOST_OPACITY = 0.22;
export const NEXT_SLOT_OPACITY = 0.5;

export const PALETTE = {
  bg: SUBSTRATE.bg,
  ground: MATERIAL.ground,
  chalk: INTENT.chalk,
  mortar: MATERIAL.mortar,
  clay: MATERIAL.clay,
  clayFallen: MATERIAL.clayFallen,
  voussoirFace: MATERIAL.clay, // a real structural voussoir is still clay, just its own shape
  hudBg: "#0C0B0A",
  hudText: SUBSTRATE.ink,
  label: SUBSTRATE.muted,
  robot: MATERIAL.steel,
  robotDark: MATERIAL.steelDark,
  robotTool: MATERIAL.machineAmber,
  cement: MATERIAL.concrete,
  stone: MATERIAL.stone,
  stoneEdge: MATERIAL.stoneEdge,
  timber: MATERIAL.timber,
  timberEdge: MATERIAL.timberEdge,
  accent: ACCENT,
  reachBand: "rgba(255,176,32,0.08)", // machine-amber, very faint - the arm's reach window
  // generic good/bad status (MetricsPanel, drawStrip's per-step reward bars) - deliberately
  // the SAME values as the measurement ramp's soft-teal/strong-red anchors below, so "good"
  // means one thing everywhere in the product, not a second unrelated green/red pair.
  good: "#6FA9AC",
  bad: "#C24A3F",
} as const;

export const HARD_BODY_COLORS: Record<string, readonly [string, string]> = {
  voussoir: [PALETTE.clay, PALETTE.mortar],
  centering: [PALETTE.timber, PALETTE.timberEdge],
  skewback: [PALETTE.stone, PALETTE.stoneEdge],
  cement: [PALETTE.cement, PALETTE.stoneEdge],
};

// atrium_sim.constants values these renderers depend on (see draw.ts/scene3d.ts for how
// each is used) - kept in one place so a constant drifting out of sync with the Python
// source is a one-line fix, not a hunt across two renderer files.
export const TOL_MM = 3.0;
export const H_MAX = 360.0;
export const COURSE_MM = 60.0;
export const ARM_MARGIN_MM = 60.0; // atrium_sim.constants.DROP_ARM_MARGIN_MM (never overridden per-run)
export const MARGIN_MM = 150.0;
export const HUD_H = 64;
// atrium_sim.arch's post-strike survival thresholds - mirrored here (not shipped in the
// replay) so the Strike page can derive its own per-arch drift/survival readout client-side
// from raw voussoir poses, the same way atrium_sim.arch.survived() does server-side.
export const SURVIVAL_DRIFT_MM = 20.0;
export const SURVIVAL_TILT_DEG = 10.0;

// --- color math -------------------------------------------------------------------------
// Everything below is pure math, deliberately hand-rolled the same way in
// atrium_sim/render/palette.py (same hash constants, same HSL formulas - NOT delegated to a
// library on one side and colorsys on the other) so a brick's jitter and an audit's tint are
// bit-for-bit the same computation whether it's drawn by pygame, the 2D canvas, or three.js.

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function hexToRgb01(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function rgb01ToHex([r, g, b]: readonly [number, number, number]): string {
  const c = (v: number) => Math.round(clamp01(v) * 255).toString(16).padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`;
}

function lerp3(
  a: readonly [number, number, number], b: readonly [number, number, number], t: number,
): [number, number, number] {
  const tt = clamp01(t);
  return [a[0] + (b[0] - a[0]) * tt, a[1] + (b[1] - a[1]) * tt, a[2] + (b[2] - a[2]) * tt];
}

// Standard HSL<->RGB, h/s/l all in [0,1) (the same convention as Python's colorsys - h is a
// FRACTION of a full turn, not degrees - so the jitter formula below ports without a unit
// conversion at the boundary).
function rgbToHsl([r, g, b]: readonly [number, number, number]): [number, number, number] {
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h: number;
  if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return [h / 6, s, l];
}

function hueToRgbChannel(p: number, q: number, tIn: number): number {
  let t = tIn;
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}

function hslToRgb([h, s, l]: readonly [number, number, number]): [number, number, number] {
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hueToRgbChannel(p, q, h + 1 / 3), hueToRgbChannel(p, q, h), hueToRgbChannel(p, q, h - 1 / 3)];
}

/** triple32, a public-domain 32-bit integer bit-mixer - deterministic and trivial to
 * replicate exactly in Python (same shifts/multiplies, masked to 32 bits), unlike a PRNG
 * library that could differ between languages. Returns a value in [0, 1). */
function hash01(n: number): number {
  let x = (n ^ 0x9e3779b9) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x / 4294967296;
}

/** A deterministic value in [-1, 1] from an integer seed - the same hash as the clay jitter
 * above, for small non-color jitter (per-course z-stagger, a brick's z-rotation) that
 * doesn't need the full HSL machinery. Use a distinct seed salt from clayJitterRgb's own
 * (brickId*3+0/1/2) so the two jitters don't accidentally correlate. */
export function jitterSigned(seed: number): number {
  return hash01(seed) * 2 - 1;
}

/** Deterministic per-brick clay variation, computed once from brick_id (not re-rolled per
 * frame - see SceneCanvas.tsx/draw.ts for where this is memoized at replay load). A wall of
 * 600 identical boxes reads as plastic; +-4deg hue / +-8% saturation / +-7% lightness is
 * enough to read as fired clay without any brick looking like a mistake. */
export function clayJitterRgb(brickId: number, baseHex: string = MATERIAL.clay): [number, number, number] {
  const dh = (hash01(brickId * 3 + 0) - 0.5) * 2 * (4 / 360);
  const ds = (hash01(brickId * 3 + 1) - 0.5) * 2 * 0.08;
  const dl = (hash01(brickId * 3 + 2) - 0.5) * 2 * 0.07;
  const [h, s, l] = rgbToHsl(hexToRgb01(baseHex));
  return hslToRgb([((h + dh) % 1 + 1) % 1, clamp01(s + ds), clamp01(l + dl)]);
}

// --- the audit's ramp (inspect mode only) ------------------------------------------------
// Signed and diverging, per the scan-vs-BIM convention (Leica Cyclone / CloudCompare-style
// deviation heatmaps): dx = brick.x - target.x is the brick's own signed lateral offset, NOT
// a material-excess/deficit read the way a surface-scan deviation is - a brick built one way
// along the wall isn't "worse" than the other, so the teal/amber split just needs to be
// CONSISTENT, not meaningful in a physical direction. Within tolerance is deliberately low-
// contrast (a neutral grey, barely nudged) - a correct wall should look calm; only the
// outliers should pull the eye.
// Exported as hex too (not just the 0..1 tuples below) so UI that isn't a three.js/canvas
// hot path - the Legend, a future colorbar, the strike page's drift readout - can reference
// the exact same anchors without re-typing the literals.
export const MEASURE_HEX = {
  neutral: "#8A857D", // 0mm - a desaturated clay-grey, not paper white
  tealSoft: "#6FA9AC", // right at -TOL_MM
  tealStrong: "#2F8990", // far under
  amberSoft: "#B99361", // right at +TOL_MM
  redStrong: "#C24A3F", // far over
} as const;
const MEASURE_NEUTRAL = hexToRgb01(MEASURE_HEX.neutral);
const MEASURE_TEAL_SOFT = hexToRgb01(MEASURE_HEX.tealSoft);
const MEASURE_TEAL_STRONG = hexToRgb01(MEASURE_HEX.tealStrong);
const MEASURE_AMBER_SOFT = hexToRgb01(MEASURE_HEX.amberSoft);
const MEASURE_RED_STRONG = hexToRgb01(MEASURE_HEX.redStrong);
export const MEASURE_FALLOFF_MM = 15.0; // distance past TOL_MM over which soft ramps to strong

function measurementRgb(dx: number, inTol: boolean): [number, number, number] {
  const ad = Math.abs(dx);
  const soft = dx >= 0 ? MEASURE_AMBER_SOFT : MEASURE_TEAL_SOFT;
  if (inTol) {
    // 0 -> TOL_MM: neutral toward soft, but capped well short of full soft saturation so an
    // in-tolerance brick never competes visually with a real outlier.
    return lerp3(MEASURE_NEUTRAL, soft, (ad / TOL_MM) * 0.6);
  }
  const strong = dx >= 0 ? MEASURE_RED_STRONG : MEASURE_TEAL_STRONG;
  return lerp3(soft, strong, (ad - TOL_MM) / MEASURE_FALLOFF_MM);
}

// --- the one function both renderers call per brick, per frame --------------------------

export type BrickStatus = "matched" | "flight" | "stray";

/** [r,g,b] in 0..1 (a three.js Color hot path - per-frame, per-brick). The single place all
 * three view modes' color rules live, so SceneCanvas.tsx and draw.ts can never quietly
 * diverge on what "inspect" or "drawing" means:
 *   - stray (toppled, unmatched): jittered clayFallen, in EVERY mode - a physical fact, not
 *     a measurement.
 *   - as-built: jittered clay for everything else - deviation is not shown at all.
 *   - inspect: matched bricks get the signed-deviation ramp; unmatched-but-upright bricks
 *     (in flight / not yet audited) get the neutral "no reading yet" grey.
 *   - drawing: flat chalk - the orthographic elevation is a line drawing, not a render. */
export function brickColorRgb(
  mode: ViewMode, brickId: number, status: BrickStatus, dx: number | null, inTol: boolean | null,
): [number, number, number] {
  if (status === "stray") return clayJitterRgb(brickId, MATERIAL.clayFallen);
  if (mode === "drawing") return hexToRgb01(INTENT.chalk);
  if (mode === "inspect") {
    if (status === "matched" && dx != null) return measurementRgb(dx, inTol ?? false);
    return MEASURE_NEUTRAL;
  }
  return clayJitterRgb(brickId, MATERIAL.clay); // as-built
}

export function brickColorHex(
  mode: ViewMode, brickId: number, status: BrickStatus, dx: number | null, inTol: boolean | null,
): string {
  return rgb01ToHex(brickColorRgb(mode, brickId, status, dx, inTol));
}

export function foldDeg(rad: number): number {
  const t = ((((rad + Math.PI / 2) % Math.PI) + Math.PI) % Math.PI) - Math.PI / 2;
  return (Math.abs(t) * 180) / Math.PI;
}

/** Brick face (w, h) in mm - kind 1 = half, everything else (0 = full) is the same size. */
export function brickFace(kind: number): [number, number] {
  return [kind === 1 ? 100 : 210, 50];
}
