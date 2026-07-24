"""Pygame renderer: ghost blueprint, quality-coloured bricks, mm HUD.

Visual language:
- dashed light outlines = unfilled blueprint targets ("ghosts"); the next
  expected slot is highlighted;
- placed bricks show a mortar-grey envelope band under a terracotta face;
- once matched, the face is tinted by placement quality: green (within ±3mm),
  amber (close), red (poor); strays go dark;
- matched bricks carry a small "+2.1" mm deviation label.
"""

from __future__ import annotations

import math

import numpy as np

from atrium_sim.blueprint import Blueprint, brick_face
from atrium_sim.constants import H_MAX, TOL_MM
from atrium_sim.physics import BrickPose

SCALE = 0.5  # px per mm
MARGIN_MM = 150.0
HUD_H = 64
Y_TOP_MM = H_MAX + 170.0  # show sky so drops/topples are visible
Y_BOT_MM = -30.0

BG = (24, 26, 32)
GROUND = (70, 72, 78)
GHOST = (110, 115, 125)
NEXT_SLOT = (240, 200, 80)
MORTAR = (105, 100, 95)
TERRACOTTA = (178, 92, 62)
STRAY = (90, 45, 40)
HUD_TEXT = (225, 225, 220)
LABEL = (200, 205, 215)
ROBOT = (90, 170, 220)        # mobile gantry body
ROBOT_DARK = (60, 120, 165)
ROBOT_TOOL = (240, 200, 80)   # the descending tool / gripper


def _quality_color(d: float, in_tol: bool) -> tuple[int, int, int]:
    if in_tol:
        return (95, 180, 90)
    # amber -> red as error grows past tolerance
    t = min(1.0, (d - TOL_MM) / 15.0)
    return (int(200 + 30 * t), int(150 * (1 - t) + 30), 40)


class PygameRenderer:
    def __init__(self, blueprint: Blueprint, render_mode: str | None):
        import pygame

        self.pygame = pygame
        self.blueprint = blueprint
        self.render_mode = render_mode
        self.w_px = int((blueprint.length + 2 * MARGIN_MM) * SCALE)
        self.h_px = HUD_H + int((Y_TOP_MM - Y_BOT_MM) * SCALE)
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
        return (x + MARGIN_MM) * SCALE, HUD_H + (Y_TOP_MM - y) * SCALE

    def _rect_corners(self, cx, cy, w, h, theta):
        c, s = math.cos(theta), math.sin(theta)
        pts = []
        for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
            pts.append(self._to_px(cx + dx * c - dy * s, cy + dx * s + dy * c))
        return pts

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
             robot: tuple | None = None) -> np.ndarray | None:
        pg = self.pygame
        surf = self.surface
        surf.fill(BG)

        # ground
        gx0, gy0 = self._to_px(-MARGIN_MM, 0)
        pg.draw.rect(
            surf, GROUND, pg.Rect(0, gy0, self.w_px, self.h_px - gy0)
        )

        # mobile robot: shaded reach window (behind everything; the body is drawn
        # on top after the bricks so it reads clearly)
        if robot is not None:
            base_x, reach = robot[0], robot[1]
            lx, _ = self._to_px(base_x - reach, 0)
            rx, _ = self._to_px(base_x + reach, 0)
            band = pg.Surface((max(1, rx - lx), self.h_px - gy0), pg.SRCALPHA)
            band.fill((240, 200, 80, 20))
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
                pg.draw.polygon(surf, NEXT_SLOT, self._rect_corners(t.x, t.y, w, h, 0.0), 2)
            else:
                self._dashed_rect(GHOST, t.x, t.y, w, h)

        # bricks
        match_by_brick = {m.brick_id: m for m in report.matches}
        stray_ids = set(report.stray_bricks)
        for p in poses:
            w, h = brick_face(p.kind)
            m = match_by_brick.get(p.brick_id)
            if m is not None:
                face = _quality_color(m.d, m.in_tol)
            elif p.brick_id in stray_ids:
                face = STRAY
            else:
                face = TERRACOTTA  # mid-fall / not yet audited
            pg.draw.polygon(surf, MORTAR, self._rect_corners(p.x, p.y, w + 9, h + 9, p.theta))
            pg.draw.polygon(surf, face, self._rect_corners(p.x, p.y, w, h, p.theta))
            if m is not None:
                label = self.font.render(f"{m.dx:+.1f}", True, LABEL)
                lx, ly = self._to_px(p.x, p.y + h / 2 + 14)
                surf.blit(label, (lx - label.get_width() / 2, ly))

        # mobile-robot body (drawn on top so it's clearly visible)
        if robot is not None:
            self._draw_robot_body(robot, poses)

        # HUD
        pg.draw.rect(surf, (18, 19, 24), pg.Rect(0, 0, self.w_px, HUD_H))
        text = "   ".join(f"{k}: {v}" for k, v in hud.items())
        surf.blit(self.hud_font.render("atrium-sim", True, NEXT_SLOT), (10, 8))
        surf.blit(self.hud_font.render(text, True, HUD_TEXT), (10, 34))

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
        pg.draw.line(surf, ROBOT, (bx - half, beam_py), (bx + half, beam_py), 5)
        # vertical mast up the middle
        pg.draw.line(surf, ROBOT, (bx, gy0 - 10), (bx, beam_py), 6)
        # wheeled chassis on the rail
        pg.draw.rect(surf, ROBOT_DARK, pg.Rect(bx - 26, gy0 - 15, 52, 15), border_radius=4)
        for wx in (bx - 15, bx + 15):
            pg.draw.circle(surf, (28, 30, 36), (wx, gy0), 6)
            pg.draw.circle(surf, ROBOT, (wx, gy0), 6, 2)

        # tool: descends from the beam to the current brick (the last-spawned pose),
        # clamped into the reach window — shows how high the release was
        if poses:
            p = poses[-1]
            tool_x = min(max(p.x, base_x - reach), base_x + reach)
            tpx, _ = self._to_px(tool_x, 0.0)
            _, tpy = self._to_px(0.0, p.y + brick_face(p.kind)[1] / 2)
            tpx, tpy = int(tpx), int(tpy)
            pg.draw.line(surf, ROBOT, (tpx, beam_py), (tpx, beam_py + 4), 8)  # trolley on beam
            pg.draw.line(surf, ROBOT_TOOL, (tpx, beam_py), (tpx, tpy), 3)      # descending tool
            pg.draw.circle(surf, ROBOT_TOOL, (tpx, tpy), 5)                    # gripper head

    def close(self):
        if self.window is not None:
            self.pygame.display.quit()
            self.window = None
