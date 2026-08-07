"""Pygame renderer: ghost blueprint, quality-coloured bricks, mm HUD.

Visual language (see atrium_sim/render/palette.py for the actual color values/formulas,
mirrored 1:1 in frontend/lib/replay/shared.ts so the GIFs and the browser renderers read as
one product, not three skins of one idea):
- dashed chalk outlines = unfilled blueprint targets ("ghosts"); the next expected slot is
  the same hue, just less translucent;
- placed bricks show a mortar-grey envelope band under a jittered-clay face;
- `mode` picks what the face color actually encodes: "as-built" (the default look - clay,
  deviation not shown at all), "inspect" (the audit's own signed-deviation ramp - what every
  GIF in this project's README has shown historically, since the README's whole story is
  about that measurement), or "drawing" (flat chalk - a line drawing, not a render);
- matched, out-of-tolerance bricks carry a small "+2.1" mm deviation label, in "inspect" mode
  only - a healthy wall should look calm, not be littered with labels on every brick.
"""

from __future__ import annotations

import math

import numpy as np

from atrium_sim.blueprint import BrickKind, Blueprint, brick_face
from atrium_sim.constants import COURSE_MM, H_MAX
from atrium_sim.physics import BrickPose
from atrium_sim.render import palette as pal

SCALE = 0.5  # px per mm
MARGIN_MM = 150.0
HUD_H = 64
Y_TOP_MM = H_MAX + 170.0  # show sky so drops/topples are visible
Y_BOT_MM = -30.0

# Ghost/next-slot targets are chalk at GHOST_OPACITY/NEXT_SLOT_OPACITY in the browser
# renderers (a real alpha channel); pygame's draw.polygon/draw.line have none, so these are
# pre-blended onto the (fixed) background color once, rather than per frame.
_GHOST_COLOR = pal.blend(pal.CHALK, pal.BG, pal.GHOST_OPACITY)
_NEXT_SLOT_COLOR = pal.blend(pal.CHALK, pal.BG, pal.NEXT_SLOT_OPACITY)


class PygameRenderer:
    def __init__(self, blueprint: Blueprint, render_mode: str | None, mode: str = "inspect"):
        import pygame

        self.pygame = pygame
        self.blueprint = blueprint
        self.render_mode = render_mode
        self.mode = mode  # "as-built" | "inspect" | "drawing" - see module docstring
        # viewport top is size-aware: tall facades (a 40-course pier, a 13-course house with
        # arched heads) must fit, not just the old fixed H_MAX (6 courses). Leaves sky headroom.
        self.y_top = max(Y_TOP_MM, blueprint.n_courses * COURSE_MM + 170.0)
        self.w_px = int((blueprint.length + 2 * MARGIN_MM) * SCALE)
        self.h_px = HUD_H + int((self.y_top - Y_BOT_MM) * SCALE)
        pygame.font.init()
        self.font = pygame.font.Font(None, 16)
        self.hud_font = pygame.font.Font(None, 20)
        self.surface = pygame.Surface((self.w_px, self.h_px))
        self.window = None
        self.clock = None
        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption("atrium-sim")
            self.window = pygame.display.set_mode((self.w_px, self.h_px))
            self.clock = pygame.time.Clock()

    # --- coordinate transform -------------------------------------------------

    def _to_px(self, x: float, y: float) -> tuple[float, float]:
        return (x + MARGIN_MM) * SCALE, HUD_H + (self.y_top - y) * SCALE

    def _rect_corners(self, cx, cy, w, h, theta):
        c, s = math.cos(theta), math.sin(theta)
        pts = []
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
            pts.append(self._to_px(cx + dx * c - dy * s, cy + dx * s + dy * c))
        return pts

    def _poly_corners(self, cx, cy, theta, local_verts):
        """Generalizes _rect_corners to an ARBITRARY canonical polygon (e.g. a tapered arch
        voussoir wedge, not just a box): rotate `local_verts` by theta, translate to (cx, cy),
        return pixel coords - same contract as _rect_corners."""
        c, s = math.cos(theta), math.sin(theta)
        return [self._to_px(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in local_verts]

    def _dashed_rect(self, color, cx, cy, w, h, dash=6):
        pg = self.pygame
        corners = self._rect_corners(cx, cy, w, h, 0.0)
        for p1, p2 in zip(corners, corners[1:] + corners[:1]):
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            n = max(1, int(length / dash))
            for i in range(0, n, 2):
                a = (p1[0] + (p2[0] - p1[0]) * i / n, p1[1] + (p2[1] - p1[1]) * i / n)
                b = (
                    p1[0] + (p2[0] - p1[0]) * min(i + 1, n) / n,
                    p1[1] + (p2[1] - p1[1]) * min(i + 1, n) / n,
                )
                pg.draw.line(self.surface, color, a, b, 1)

    # --- main draw --------------------------------------------------------------

    def draw(self, poses: list[BrickPose], report, hud: dict, cursor: int,
              robot: tuple | None = None, hard_bodies: list | None = None) -> np.ndarray | None:
        pg = self.pygame
        surf = self.surface
        surf.fill(pal.BG)

        # ground
        gx0, gy0 = self._to_px(-MARGIN_MM, 0)
        pg.draw.rect(
            surf, pal.GROUND, pg.Rect(0, gy0, self.w_px, self.h_px - gy0)
        )

        # mobile robot: shaded reach window (behind everything; the body is drawn
        # on top after the bricks so it reads clearly)
        if robot is not None:
            base_x, reach = robot[0], robot[1]
            lx, _ = self._to_px(base_x - reach, 0)
            rx, _ = self._to_px(base_x + reach, 0)
            band = pg.Surface((max(1, rx - lx), self.h_px - gy0), pg.SRCALPHA)
            band.fill((*pal.ROBOT_TOOL, 20))
            surf.blit(band, (lx, gy0))

        # ghost blueprint + next expected slot
        matched_ids = {m.target_id for m in report.matches}
        open_in_cursor = [
            t for t in self.blueprint.course_targets(cursor) if t.tid not in matched_ids
        ]
        next_tid = min((t.tid for t in sorted(open_in_cursor, key=lambda t: t.x)), default=None)
        for t in self.blueprint.targets:
            if t.tid in matched_ids:
                continue
            w, h = brick_face(t.kind)
            if t.tid == next_tid:
                pg.draw.polygon(surf, _NEXT_SLOT_COLOR, self._rect_corners(t.x, t.y, w, h, 0.0), 2)
            else:
                self._dashed_rect(_GHOST_COLOR, t.x, t.y, w, h)

        # static hard bodies: lintels/sills/cement heads (permanent) and the arch centering/
        # skewback (centering is temporary - visible only until the ring closes and it's struck;
        # "voussoir" here is the older cosmetic lintel_soldier style's sensor fringe, kept
        # distinct from the real dynamic VOUSSOIR bricks drawn below). "drawing" mode flattens
        # every material to chalk - an elevation is a line drawing, not a render.
        for _sid, kind, verts in (hard_bodies or []):
            pts = [self._to_px(x, y) for x, y in verts]
            if len(pts) < 3:
                continue
            if self.mode == "drawing":
                fill, edge = pal.CHALK, pal.CHALK
            elif kind == "voussoir":
                fill, edge = pal.CLAY, pal.MORTAR
            elif kind == "centering":
                fill, edge = pal.TIMBER, pal.TIMBER_EDGE
            elif kind == "skewback":
                fill, edge = pal.STONE, pal.STONE_EDGE
            elif kind == "cement":
                fill, edge = pal.CEMENT, pal.STONE_EDGE
            else:
                fill, edge = pal.STONE, pal.STONE_EDGE
            pg.draw.polygon(surf, fill, pts)
            pg.draw.polygon(surf, edge, pts, 2)

        # bricks
        match_by_brick = {m.brick_id: m for m in report.matches}
        stray_ids = set(report.stray_bricks)
        for p in poses:
            if p.kind == BrickKind.VOUSSOIR and p.verts:
                # a real structural arch wedge: arbitrary polygon, not scored by the flat-wall
                # audit - "flight" status is the honest read in every mode (see palette.py's
                # brick_color / shared.ts's brickColorRgb for the shared rule table).
                pts = self._poly_corners(p.x, p.y, p.theta, p.verts)
                if len(pts) >= 3:
                    face = pal.brick_color(self.mode, p.brick_id, "flight", None, None)
                    pg.draw.polygon(surf, face, pts)
                    pg.draw.polygon(surf, pal.MORTAR, pts, 2)
                continue
            w, h = brick_face(p.kind)
            m = match_by_brick.get(p.brick_id)
            if m is not None:
                status, dx, in_tol = "matched", m.dx, m.in_tol
            elif p.brick_id in stray_ids:
                status, dx, in_tol = "stray", None, None
            else:
                status, dx, in_tol = "flight", None, None
            face = pal.brick_color(self.mode, p.brick_id, status, dx, in_tol)
            pg.draw.polygon(surf, pal.MORTAR, self._rect_corners(p.x, p.y, w + 9, h + 9, p.theta))
            pg.draw.polygon(surf, face, self._rect_corners(p.x, p.y, w, h, p.theta))
            if m is not None and not m.in_tol and self.mode == "inspect":
                label = self.font.render(f"{m.dx:+.1f}", True, pal.LABEL)
                lx, ly = self._to_px(p.x, p.y + h / 2 + 14)
                surf.blit(label, (lx - label.get_width() / 2, ly))

        # mobile-robot body (drawn on top so it's clearly visible)
        if robot is not None:
            self._draw_robot_body(robot, poses)

        # HUD
        pg.draw.rect(surf, pal.HUD_BG, pg.Rect(0, 0, self.w_px, HUD_H))
        text = "   ".join(f"{k}: {v}" for k, v in hud.items()) + f"   view: {self.mode}"
        surf.blit(self.hud_font.render("atrium-sim", True, pal.ACCENT), (10, 8))
        surf.blit(self.hud_font.render(text, True, pal.HUD_TEXT), (10, 34))

        if self.render_mode == "human":
            self.window.blit(surf, (0, 0))
            pg.event.pump()
            pg.display.flip()
            self.clock.tick(60)

        frame = np.transpose(pg.surfarray.array3d(surf), (1, 0, 2))
        return frame

    def _draw_robot_body(self, robot: tuple, poses: list[BrickPose]) -> None:
        """A mobile gantry: wheeled chassis on the rail, a mast, a top beam spanning
        the reach, and a tool that descends from the beam to the brick being placed
        (so a high release visibly drops from up top)."""
        pg = self.pygame
        surf = self.surface
        base_x, reach = robot[0], robot[1]
        gantry_y = robot[2] if len(robot) > 2 and robot[2] else (H_MAX + 60.0)

        bx, gy0 = self._to_px(base_x, 0.0)
        _, beam_py = self._to_px(base_x, gantry_y)
        bx, gy0, beam_py = int(bx), int(gy0), int(beam_py)

        # top beam spanning the reach window
        half = int(reach * SCALE)
        pg.draw.line(surf, pal.ROBOT, (bx - half, beam_py), (bx + half, beam_py), 5)
        # vertical mast up the middle
        pg.draw.line(surf, pal.ROBOT, (bx, gy0 - 10), (bx, beam_py), 6)
        # wheeled chassis on the rail
        pg.draw.rect(surf, pal.ROBOT_DARK, pg.Rect(bx - 26, gy0 - 15, 52, 15), border_radius=4)
        for wx in (bx - 15, bx + 15):
            pg.draw.circle(surf, pal.BG, (wx, gy0), 6)
            pg.draw.circle(surf, pal.ROBOT, (wx, gy0), 6, 2)

        # tool: descends from the beam to the current brick (the last-spawned pose),
        # clamped into the reach window — shows how high the release was
        if poses:
            p = poses[-1]
            tool_x = min(max(p.x, base_x - reach), base_x + reach)
            tpx, _ = self._to_px(tool_x, 0.0)
            _, tpy = self._to_px(0.0, p.y + brick_face(p.kind)[1] / 2)
            tpx, tpy = int(tpx), int(tpy)
            pg.draw.line(surf, pal.ROBOT, (tpx, beam_py), (tpx, beam_py + 4), 8)  # trolley on beam
            pg.draw.line(surf, pal.ROBOT_TOOL, (tpx, beam_py), (tpx, tpy), 3)      # descending tool
            pg.draw.circle(surf, pal.ROBOT_TOOL, (tpx, tpy), 5)                    # gripper head

    def close(self):
        if self.window is not None:
            self.pygame.display.quit()
            self.window = None
