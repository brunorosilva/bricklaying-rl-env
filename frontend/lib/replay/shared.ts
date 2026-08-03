// Palette + pure geometry/color helpers shared by BOTH replay renderers (draw.ts's 2D
// canvas port and scene3d/SceneCanvas's 3D scene) - one source of truth for "what does
// atrium_sim/render/renderer.py's palette and quality-tint formula actually say", so the
// two renderers can never silently drift into looking like different products.

export const PALETTE = {
  bg: "#181a20",
  ground: "#46484e",
  ghost: "#6e737d",
  nextSlot: "#f0c850",
  mortar: "#69645f",
  terracotta: "#b25c3e",
  stray: "#5a2d28",
  hudBg: "#12131a",
  hudText: "#e1e1dc",
  label: "#c8cdd7",
  robot: "#5aaadc",
  robotDark: "#3c78a5",
  robotTool: "#f0c850",
  cement: "#96948f",
  stone: "#b2aca0",
  stoneEdge: "#6e6c68",
  centering: "#785f3c",
  centeringEdge: "#503c23",
  voussoirFace: "#b25c3e",
  reachBand: "rgba(240,200,80,0.10)",
} as const;

export const HARD_BODY_COLORS: Record<string, readonly [string, string]> = {
  voussoir: [PALETTE.terracotta, PALETTE.mortar],
  centering: [PALETTE.centering, PALETTE.centeringEdge],
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

export function qualityColor(d: number, inTol: boolean): string {
  if (inTol) return "#5fb45a";
  const t = Math.min(1, (d - TOL_MM) / 15);
  return `rgb(${Math.round(200 + 30 * t)},${Math.round(150 * (1 - t) + 30)},40)`;
}

/** [r,g,b] in 0..1, for three.js Color - same formula as qualityColor, without the
 * string round-trip (a per-frame, per-brick hot path in the 3D renderer). */
export function qualityColorRgb(d: number, inTol: boolean): [number, number, number] {
  if (inTol) return [95 / 255, 180 / 255, 90 / 255];
  const t = Math.min(1, (d - TOL_MM) / 15);
  return [(200 + 30 * t) / 255, (150 * (1 - t) + 30) / 255, 40 / 255];
}

export function foldDeg(rad: number): number {
  const t = ((((rad + Math.PI / 2) % Math.PI) + Math.PI) % Math.PI) - Math.PI / 2;
  return (Math.abs(t) * 180) / Math.PI;
}

/** Brick face (w, h) in mm - kind 1 = half, everything else (0 = full) is the same size. */
export function brickFace(kind: number): [number, number] {
  return [kind === 1 ? 100 : 210, 50];
}
