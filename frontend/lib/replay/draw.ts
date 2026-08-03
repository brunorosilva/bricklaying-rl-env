// Canvas replay renderer - a faithful, DPR-aware port of atrium_sim/render/renderer.py's
// PygameRenderer, so the browser and the server-rendered GIFs read as the same product.
// Callers are responsible for DPR scaling (see StageCanvas): set canvas.width/height to
// cssSize*devicePixelRatio, call ctx.scale(dpr, dpr) once, then everything here works in
// plain CSS-pixel coordinates.

import type { Brick, Frame, HardBody, Match, Replay, Target, View } from "./types";
import {
  ARM_MARGIN_MM, COURSE_MM, H_MAX, HARD_BODY_COLORS, HUD_H, MARGIN_MM,
  PALETTE, foldDeg, qualityColor,
} from "./shared";

// palette aliases (exact atrium_sim/render/renderer.py values, see shared.ts) - kept as
// short local names so the drawing code below reads the same as before the extraction
const BG = PALETTE.bg;
const GROUND = PALETTE.ground;
const GHOST = PALETTE.ghost;
const NEXT_SLOT = PALETTE.nextSlot;
const MORTAR = PALETTE.mortar;
const TERRACOTTA = PALETTE.terracotta;
const STRAY = PALETTE.stray;
const HUD_BG = PALETTE.hudBg;
const HUD_TEXT = PALETTE.hudText;
const LABEL = PALETTE.label;
const ROBOT = PALETTE.robot;
const ROBOT_DARK = PALETTE.robotDark;
const ROBOT_TOOL = PALETTE.robotTool;
const STONE = PALETTE.stone;
const STONE_EDGE = PALETTE.stoneEdge;
const VOUSSOIR_FACE = PALETTE.voussoirFace;
const REACH_BAND = PALETTE.reachBand;

export function computeView(lengthMm: number, nCourses: number, cssW: number, cssH: number): View {
  const xmin = -MARGIN_MM;
  const xmax = lengthMm + MARGIN_MM;
  const ymin = -30;
  // size-aware sky headroom - mirrors renderer.py's `max(Y_TOP_MM, n_courses*COURSE_MM+170)`
  const ymax = Math.max(H_MAX + 170, nCourses * 60 + 170);
  const worldW = xmax - xmin;
  const worldH = ymax - ymin;
  const availH = cssH - HUD_H;
  const s = Math.min(cssW / worldW, availH / worldH);
  return { xmin, ymax, s, ox: (cssW - worldW * s) / 2, oy: (availH - worldH * s) / 2 };
}

const px = (v: View, x: number) => v.ox + (x - v.xmin) * v.s;
const py = (v: View, y: number) => HUD_H + v.oy + (v.ymax - y) * v.s;

type Pt = [number, number];

function rectCorners(v: View, cx: number, cy: number, w: number, h: number, theta: number): Pt[] {
  const c = Math.cos(theta), s = Math.sin(theta);
  return ([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]] as const).map(
    ([dx, dy]) => [px(v, cx + dx * c - dy * s), py(v, cy + dx * s + dy * c)] as Pt,
  );
}

/** Rotate+translate arbitrary LOCAL (pre-rotation) polygon points by (cx, cy, theta) - the
 * voussoir path, generalizing rectCorners the same way renderer.py's _poly_corners does. */
function polyCorners(v: View, cx: number, cy: number, theta: number, localVerts: [number, number][]): Pt[] {
  const c = Math.cos(theta), s = Math.sin(theta);
  return localVerts.map(([lx, ly]) => [px(v, cx + lx * c - ly * s), py(v, cy + lx * s + ly * c)] as Pt);
}

function fillPoly(ctx: CanvasRenderingContext2D, pts: Pt[], fill: string) {
  if (pts.length < 3) return;
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

function strokePoly(ctx: CanvasRenderingContext2D, pts: Pt[], stroke: string, width = 2) {
  if (pts.length < 2) return;
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  ctx.closePath();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.stroke();
}

function dashedRect(ctx: CanvasRenderingContext2D, v: View, color: string, cx: number, cy: number, w: number, h: number) {
  ctx.save();
  ctx.setLineDash([5, 4]);
  strokePoly(ctx, rectCorners(v, cx, cy, w, h, 0), color, 1);
  ctx.restore();
}

// --- the mobile gantry (mast, beam, wheeled chassis, descending tool) -------------------
// A faithful port of renderer.py's _draw_robot_body - the frontend previously drew only a
// flat triangle marker, so arch/robot GIFs and the browser replay looked like different
// products even for the identical episode.
function drawGantry(
  ctx: CanvasRenderingContext2D, v: View, baseX: number, reach: number,
  nCourses: number, lastBrick: Brick | undefined,
) {
  const bx = px(v, baseX);
  const gy0 = py(v, 0);
  // arm "home" height - the SAME size-dependent formula robot_env._render_frame passes to
  // the GIF renderer (COURSE_MM*n_courses + arm_margin_mm), not a fixed constant: a fixed
  // beam height would sit UNDER the wall top on anything taller than ~5 courses.
  const beamY = py(v, COURSE_MM * nCourses + ARM_MARGIN_MM);
  const half = reach * v.s;

  ctx.lineCap = "round";
  ctx.strokeStyle = ROBOT;
  ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(bx - half, beamY); ctx.lineTo(bx + half, beamY); ctx.stroke(); // top beam
  ctx.lineWidth = 6;
  ctx.beginPath(); ctx.moveTo(bx, gy0 - 10); ctx.lineTo(bx, beamY); ctx.stroke(); // mast

  // wheeled chassis
  const chW = 52, chH = 15, r = 4;
  ctx.fillStyle = ROBOT_DARK;
  ctx.beginPath();
  ctx.roundRect(bx - chW / 2, gy0 - chH, chW, chH, r);
  ctx.fill();
  for (const wx of [bx - 15, bx + 15]) {
    ctx.beginPath(); ctx.arc(wx, gy0, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#1c1e24"; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = ROBOT; ctx.stroke();
  }

  // tool: descends from the beam to the last-spawned brick, clamped into the reach window -
  // shows how high the release was
  if (lastBrick) {
    const brickX = Math.min(Math.max(lastBrick[0], baseX - reach), baseX + reach);
    const h = lastBrick[3] === 1 ? 50 : lastBrick[3] === 2 ? 0 : 50; // half-brick face height == full's
    const toolX = px(v, brickX);
    const toolY = py(v, lastBrick[1] + h / 2 + 0); // brick top edge
    ctx.lineCap = "butt";
    ctx.lineWidth = 8; ctx.strokeStyle = ROBOT;
    ctx.beginPath(); ctx.moveTo(toolX, beamY); ctx.lineTo(toolX, beamY + 4); ctx.stroke(); // trolley
    ctx.lineWidth = 3; ctx.strokeStyle = ROBOT_TOOL;
    ctx.beginPath(); ctx.moveTo(toolX, beamY); ctx.lineTo(toolX, toolY); ctx.stroke(); // descending tool
    ctx.beginPath(); ctx.arc(toolX, toolY, 5, 0, Math.PI * 2); ctx.fillStyle = ROBOT_TOOL; ctx.fill(); // gripper
  }
  ctx.lineCap = "butt";
}

export type DrawOpts = { labels: boolean };

export function drawScene(ctx: CanvasRenderingContext2D, cssW: number, cssH: number, v: View, replay: Replay, frame: Frame, opts: DrawOpts) {
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, cssW, cssH);

  const groundY = py(v, 0);
  ctx.fillStyle = GROUND;
  ctx.fillRect(0, groundY, cssW, cssH - groundY);

  // mobile-robot reach band (behind everything; the body is drawn on top after the bricks)
  const isRobot = !!replay.robot;
  if (isRobot && frame.base != null) {
    const reach = replay.robot!.reach;
    const lx = px(v, frame.base - reach), rx = px(v, frame.base + reach);
    ctx.fillStyle = REACH_BAND;
    ctx.fillRect(lx, groundY, rx - lx, cssH - groundY);
  }

  const matches: Match[] = frame.st.matches ?? [];
  const matchByTarget = new Map<number, Match>();
  const matchByBrick = new Map<number, Match>();
  for (const m of matches) { matchByTarget.set(m.target_id, m); matchByBrick.set(m.brick_id, m); }

  // ghost blueprint + the next expected slot (cursor's course, leftmost unfilled)
  const cursor = frame.st.cursor;
  let nextTarget: Target | null = null;
  if (cursor != null) {
    for (const t of replay.targets) {
      if (t.course !== cursor || matchByTarget.has(t.tid)) continue;
      if (!nextTarget || t.x < nextTarget.x) nextTarget = t;
    }
  }
  for (const t of replay.targets) {
    if (matchByTarget.has(t.tid)) continue;
    if (t === nextTarget) {
      strokePoly(ctx, rectCorners(v, t.x, t.y, t.w, t.h, 0), NEXT_SLOT, 2);
    } else {
      dashedRect(ctx, v, GHOST, t.x, t.y, t.w, t.h);
    }
  }

  // static hard bodies (lintels/sills/cement heads, arch centering/skewback), faded in by
  // the frame index at which they first appear
  for (const hb of replay.hard_bodies ?? ([] as HardBody[])) {
    if (hb.appear > frame.gi) continue;
    const pts = hb.verts.map(([x, y]) => [px(v, x), py(v, y)] as Pt);
    const [fill, edge] = HARD_BODY_COLORS[hb.kind] ?? [STONE, STONE_EDGE];
    fillPoly(ctx, pts, fill);
    strokePoly(ctx, pts, edge, 2);
  }

  // bricks
  ctx.font = `${Math.max(9, 11 * v.s * 2)}px var(--font-mono), monospace`;
  ctx.textAlign = "center";
  for (const b of frame.bricks) {
    const [x, y, theta, kind, brickId] = b;
    if (kind === 2 && b.length === 6) {
      // real structural arch wedge: arbitrary polygon, not scored by the flat-wall audit -
      // its own distinct material, no quality tint/label (matches renderer.py exactly)
      const pts = polyCorners(v, x, y, theta, b[5]);
      fillPoly(ctx, pts, VOUSSOIR_FACE);
      strokePoly(ctx, pts, MORTAR, 2);
      continue;
    }
    const w = kind === 1 ? 100 : 210, h = 50;
    const m = matchByBrick.get(brickId);
    let face: string = STRAY;
    if (m) face = qualityColor(m.d, m.in_tol);
    else if (foldDeg(theta) < 15) face = TERRACOTTA; // upright, in flight / not yet audited
    fillPoly(ctx, rectCorners(v, x, y, w + 9, h + 9, theta), MORTAR);
    fillPoly(ctx, rectCorners(v, x, y, w, h, theta), face);
    if (opts.labels && m) {
      ctx.fillStyle = LABEL;
      ctx.fillText(`${m.dx >= 0 ? "+" : ""}${m.dx.toFixed(1)}`, px(v, x), py(v, y + h / 2 + 14));
    }
  }

  if (isRobot && frame.base != null) {
    drawGantry(ctx, v, frame.base, replay.robot!.reach, replay.n_courses, frame.bricks[frame.bricks.length - 1]);
  }

  // HUD
  ctx.fillStyle = HUD_BG;
  ctx.fillRect(0, 0, cssW, HUD_H);
  ctx.textAlign = "left";
  ctx.font = "13px var(--font-mono), monospace";
  ctx.fillStyle = NEXT_SLOT;
  ctx.fillText("atrium-sim", 12, 20);
  ctx.fillStyle = HUD_TEXT;
  const st = frame.st;
  const tail = isRobot ? `moves ${st.moves ?? 0}   placed ${st.placements ?? 0}` : `waste ${st.waste ?? 0}`;
  const mode = isRobot ? `${["place", "move ←", "move →"][st.mode ?? 0]}   ` : "";
  ctx.fillText(
    `${replay._policy ?? ""}   ${replay.spec.n_modules}m × ${replay.spec.n_courses}c   ` +
      `step ${st.i + 1}/${replay.steps.length}   ${mode}in-tol ${(st.frac_in_tol * 100).toFixed(0)}%   ` +
      `filled ${(st.frac_filled * 100).toFixed(0)}%   ${tail}   return ${st.return.toFixed(2)}`,
    12,
    44,
  );
}

export function drawStrip(ctx: CanvasRenderingContext2D, cssW: number, cssH: number, replay: Replay, curStep: number) {
  ctx.clearRect(0, 0, cssW, cssH);
  const steps = replay.steps, n = steps.length, bw = cssW / n, mid = cssH * 0.55;
  const maxAbs = Math.max(0.5, ...steps.map((s) => Math.abs(s.reward)));
  ctx.strokeStyle = "#323847";
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(cssW, mid); ctx.stroke();
  for (let i = 0; i < n; i++) {
    const r = steps[i].reward;
    const hgt = (Math.abs(r) / maxAbs) * (cssH * 0.42);
    ctx.fillStyle = i === curStep ? "#f0c850" : r >= 0 ? "#5fb45a" : "#d64b45";
    ctx.fillRect(i * bw + 0.5, r >= 0 ? mid - hgt : mid, Math.max(1, bw - 1), hgt);
  }
}
