"""BrickLayerEnv: the Gymnasium environment tying blueprint, physics and audit together.

One step = one brick. The agent controls exactly two things:
  a[0] -> x-position along the wall (continuous, mm - the ±3mm game)
  a[1] -> brick kind by sign (FULL if < 0 else HALF)
The row is automatic (lowest course with an unmatched target), rotation and
drop height are env-owned, and there is no STOP action.

Reward is potential-based over the audit (see reward.py). Every terminal path
runs an extra final settle and re-audits BEFORE terminal terms are granted, so
"shove the wobbling wall and quit", "sag through the finish line" and
"slow-collapse past the settle cap" are all charged, not rewarded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from atrium_sim import observations
from atrium_sim.blueprint import (
    Blueprint,
    BrickKind,
    WallSpec,
    brick_face,
    generate_blueprint,
    sample_spec,
)
from atrium_sim.constants import (
    FINAL_SETTLE_SUBSTEPS,
    MAX_SETTLE_SUBSTEPS,
    OFFSET_RANGE_MM,
    OVERHANG_MM,
)
from atrium_sim.physics import PhysicsWorld
from atrium_sim.reward import AuditReport, RewardConfig, audit, potential


@dataclass(frozen=True)
class EnvConfig:
    max_settle_substeps: int = MAX_SETTLE_SUBSTEPS
    final_settle_substeps: int = FINAL_SETTLE_SUBSTEPS
    overhang_mm: float = OVERHANG_MM
    suite: str = "train"
    # "slot_relative": a[0] nudges +-offset_range_mm around the env's next open
    # slot (the env owns slot SELECTION; the agent owns precision + brick kind).
    # "absolute": a[0] spans the whole wall (the harder, ~unlearnable variant;
    # kept for research comparison).
    action_mode: str = "slot_relative"
    offset_range_mm: float = OFFSET_RANGE_MM
    # When True, a collapse (>=3 matched bricks knocked loose in one settle) ends
    # the episode with a flat penalty. Set False to let the agent recover and keep
    # building - useful early in training so horizontal exploration isn't punished
    # into a safe-but-useless "single column" local optimum. Topples still cost via
    # the natural potential drop either way.
    collapse_terminal: bool = True


class BrickLayerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        env_cfg: EnvConfig | None = None,
        reward_cfg: RewardConfig | None = None,
    ):
        self.env_cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(observations.OBS_DIM,), dtype=np.float32
        )

        self._renderer = None
        self.frame_sink: list | None = None  # recorder hook: rgb frames incl. settle frames
        self.tick_callback = None  # webviz hook: called each captured substep during settling

        # episode state (set in reset)
        self.blueprint: Blueprint | None = None
        self.world: PhysicsWorld | None = None
        self.report: AuditReport | None = None

    # --- gym API --------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        spec: WallSpec = (options or {}).get("spec") or sample_spec(
            self.np_random, self.env_cfg.suite
        )
        self.blueprint = generate_blueprint(spec)
        self.world = PhysicsWorld(self.blueprint.length)  # fresh Space, no hidden solver state
        self.budget = self.blueprint.budget
        self.placements = 0
        self.halves_used = 0
        self.off_canvas = 0
        self.cursor = 0
        self.last_disturbance = 0.0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self._step_costs = 0.0
        self._terminal_terms = 0.0
        # optional scenario: pre-place some bricks so the agent completes a
        # partial wall instead of starting from bare ground
        self._prefill(self._scenario_targets((options or {}).get("scenario")))
        self.report = self._audit()
        self.cursor = self._lowest_open_course()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), self._info(terminal=False)

    def _scenario_targets(self, scenario: str | None) -> list[int]:
        """Which blueprint targets to pre-place for a named scenario."""
        if not scenario or scenario == "empty":
            return []
        bp = self.blueprint
        # Only physically-stable prefills: bricks need support from below, so a
        # sparse/checkerboard prefill just topples. These all leave a solid base.
        if scenario == "prefill_base":  # bottom half already built; finish the top
            half = bp.n_courses // 2
            return [t.tid for t in bp.targets if t.course < half]
        if scenario == "almost":  # everything but the top course; lay the final row
            top = bp.n_courses - 1
            return [t.tid for t in bp.targets if t.course < top]
        if scenario == "top_gaps":  # solid wall minus a few holes in the top row
            top = bp.n_courses - 1
            top_ids = [t.tid for t in bp.targets if t.course == top]
            n_holes = min(3, len(top_ids))
            holes = set(
                self.np_random.choice(top_ids, size=n_holes, replace=False).tolist()
            ) if top_ids else set()
            return [t.tid for t in bp.targets if t.tid not in holes]
        raise ValueError(f"unknown scenario: {scenario!r}")

    def _prefill(self, target_ids: list[int]) -> None:
        """Place the given targets exactly, bottom-up (tid order), settling each.

        Prefilled bricks do not count as agent placements/cuts.
        """
        ids = set(target_ids)
        for t in self.blueprint.targets:  # tid order is course-major -> bottom-up
            if t.tid in ids:
                self.world.spawn_brick(t.x, t.kind, t.course)
                self.world.settle(self.env_cfg.max_settle_substeps)

    def step(self, action):
        assert self.world is not None, "call reset() first"
        cfg = self.reward_cfg
        n = self.blueprint.n_targets
        prev_report = self.report
        prev_matched = len(prev_report.matches)

        # 1. decode and place
        x, kind = self._decode(np.asarray(action, dtype=np.float64))
        if kind == BrickKind.HALF:
            self.halves_used += 1
        self.placements += 1
        pre_positions = self.world.positions()
        brick_id = self.world.spawn_brick(x, kind, self.cursor)
        if brick_id is None:
            self.off_canvas += 1  # nowhere to spawn: placement burned, stray-equivalent waste

        # 2. settle and audit
        removed = self._settle(self.env_cfg.max_settle_substeps)
        self.off_canvas += len(removed)
        self.report = self._audit()
        self.last_disturbance = self._disturbance(pre_positions, brick_id)

        reward = potential(self.report, cfg) - potential(prev_report, cfg)
        step_cost = cfg.c_step_frac * cfg.r_scale / n
        reward -= step_cost
        self._step_costs += step_cost

        # 3. termination - every terminal path final-settles and re-audits first
        terminated = False
        dropped = prev_matched - len(self.report.matches)
        collapse = dropped >= max(cfg.collapse_min_bricks, math.ceil(cfg.collapse_frac * prev_matched))
        collapse_end = collapse and self.env_cfg.collapse_terminal
        complete = not self.report.missing_targets
        budget_out = self.placements >= self.budget

        if collapse_end or complete or budget_out:
            reward += self._final_settle_delta()
            complete = not self.report.missing_targets
            if collapse_end:
                reward -= cfg.collapse_penalty
                self._terminal_terms -= cfg.collapse_penalty
                terminated = True
            elif complete:
                bonus = cfg.bonus_fill
                if self.report.frac_in_tol == 1.0:
                    bonus += cfg.bonus_perfect
                reward += bonus
                self._terminal_terms += bonus
                terminated = True
            elif budget_out:
                terminated = True
            # else: the final settle knocked a brick out and budget remains -> play on

        # 4. cursor = lowest course with an unmatched target (retries after topples work)
        self.cursor = self._lowest_open_course()

        self.last_reward = float(reward)
        self.episode_return += float(reward)
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), float(reward), terminated, False, self._info(terminal=terminated)

    # --- internals --------------------------------------------------------------

    def _next_open_target(self):
        """Leftmost unmatched target in the cursor course (the slot to place next)."""
        matched = {m.target_id for m in self.report.matches}
        open_t = [t for t in self.blueprint.course_targets(self.cursor) if t.tid not in matched]
        return min(open_t, key=lambda t: t.x) if open_t else None

    def _decode(self, a: np.ndarray) -> tuple[float, BrickKind]:
        kind = BrickKind.FULL if a[1] < 0 else BrickKind.HALF
        w = brick_face(kind)[0]
        lo = w / 2.0 - self.env_cfg.overhang_mm
        hi = self.blueprint.length - w / 2.0 + self.env_cfg.overhang_mm
        if self.env_cfg.action_mode == "slot_relative":
            slot = self._next_open_target()
            base = slot.x if slot is not None else self.blueprint.length / 2.0
            x = base + float(a[0]) * self.env_cfg.offset_range_mm
        else:  # absolute
            x = (float(a[0]) + 1.0) / 2.0 * self.blueprint.length
        return float(np.clip(x, lo, hi)), kind

    def _audit(self) -> AuditReport:
        return audit(
            self.world.poses(),
            self.blueprint,
            self.reward_cfg,
            off_canvas=self.off_canvas,
            halves_used=self.halves_used,
        )

    def _settle(self, substeps: int) -> list[int]:
        cbs = []
        if self.render_mode == "human" or self.frame_sink is not None:
            cbs.append(self._render_frame)
        if self.tick_callback is not None:
            cbs.append(self.tick_callback)
        if not cbs:
            frame_cb = None
        elif len(cbs) == 1:
            frame_cb = cbs[0]
        else:
            frame_cb = lambda: [cb() for cb in cbs]  # noqa: E731
        used, removed = self.world.settle(substeps, frame_cb=frame_cb)
        return removed

    def _final_settle_delta(self) -> float:
        """Extra settle + re-audit before ANY terminal decision (anti slow-collapse)."""
        before = potential(self.report, self.reward_cfg)
        removed = self._settle(self.env_cfg.final_settle_substeps)
        self.off_canvas += len(removed)
        self.report = self._audit()
        return potential(self.report, self.reward_cfg) - before

    def _disturbance(self, pre: dict[int, tuple[float, float]], new_id: int | None) -> float:
        post = self.world.positions()
        moved = [
            math.hypot(post[bid][0] - x, post[bid][1] - y)
            for bid, (x, y) in pre.items()
            if bid in post and bid != new_id
        ]
        return max(moved, default=0.0)

    def _lowest_open_course(self) -> int:
        target_course = {t.tid: t.course for t in self.blueprint.targets}
        open_courses = [target_course[tid] for tid in self.report.missing_targets]
        return min(open_courses, default=self.blueprint.n_courses - 1)

    def _obs(self) -> np.ndarray:
        course = self.blueprint.course_targets(self.cursor)
        matched_ids = {m.target_id for m in self.report.matches}
        matched_in_course = sum(1 for t in course if t.tid in matched_ids)
        open_t = [t for t in course if t.tid not in matched_ids]
        next_slot = min(open_t, key=lambda t: t.x) if open_t else None
        g = observations.GlobalState(
            cursor=self.cursor,
            course_fill_frac=matched_in_course / len(course),
            bricks_left=max(0, self.budget - self.placements),
            budget=self.budget,
            cuts=self.halves_used,
            n_strays=len(self.report.stray_bricks),
            last_disturbance_mm=self.last_disturbance,
            next_slot_x=next_slot.x if next_slot else 0.0,
            next_slot_is_half=1.0 if (next_slot and next_slot.kind == BrickKind.HALF) else 0.0,
        )
        return observations.encode(self.blueprint, self.report, g)

    def _info(self, terminal: bool) -> dict[str, Any]:
        r = self.report
        info: dict[str, Any] = {
            "frac_in_tol": r.frac_in_tol,
            "frac_filled": r.frac_filled,
            "waste_count": r.waste_count,
            "score": r.score,
            "placements": self.placements,
        }
        if terminal:
            # flat picklable dict: the published metrics + return decomposition
            info["metrics"] = {
                "frac_in_tol": r.frac_in_tol,
                "frac_filled": r.frac_filled,
                "waste_frac": r.waste_frac,
                "waste_count": float(r.waste_count),
                "mean_abs_dev_mm": r.mean_abs_dev_mm,
                "p95_abs_dev_mm": r.p95_abs_dev_mm,
                "bond_violations": float(r.bond_violations),
                "plumb_dev_mm": r.plumb_dev_mm,
                "score": r.score,
                "completed": float(r.frac_filled == 1.0),
                "final_potential": potential(r, self.reward_cfg),
                "terminal_terms": self._terminal_terms,
                "step_costs": self._step_costs,
                "episode_return": self.episode_return,
                "placements": float(self.placements),
                "halves_used": float(self.halves_used),
            }
        return info

    # --- rendering --------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self):
        if self._renderer is None:
            from atrium_sim.render.renderer import PygameRenderer

            self._renderer = PygameRenderer(self.blueprint, self.render_mode)
        frame = self._renderer.draw(
            poses=self.world.poses(),
            report=self.report,
            hud={
                "spec": f"{self.blueprint.spec.n_modules}m x {self.blueprint.spec.n_courses}c",
                "course": self.cursor,
                "bricks left": max(0, self.budget - self.placements),
                "cuts": self.halves_used,
                "strays": len(self.report.stray_bricks),
                "in tol": f"{self.report.frac_in_tol:.0%}",
                "reward": f"{self.last_reward:+.3f}",
                "return": f"{self.episode_return:+.2f}",
            },
            cursor=self.cursor,
        )
        if self.frame_sink is not None and frame is not None:
            self.frame_sink.append(frame)
        return frame

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
