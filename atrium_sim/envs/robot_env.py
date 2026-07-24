"""BrickLayerRobot-v0: a mobile bricklaying robot with finite reach.

This is the "hard problem" variant. The robot sits on a rail at position
`base_x` and can only place bricks whose target is within REACH_MM of the base.
Because REACH is smaller than a typical wall, the robot MUST move to finish -
and moving costs and earns nothing directly. So on top of the (learnable)
precision layer, the agent has to learn a movement/coverage strategy under
delayed reward: clear what you can reach, then move to uncovered wall, without
dithering.

Action is HYBRID (this is what makes "move or not" a real decision):
    (mode in {PLACE, MOVE_LEFT, MOVE_RIGHT},  [offset, kind])
- PLACE: lay a brick at the nearest in-reach open slot, nudged by offset, of the
  chosen kind. If nothing is reachable, it's a wasted step (small penalty).
- MOVE_LEFT / MOVE_RIGHT: shift the base by one module; costs a little.

Reward reuses the same pure audit potential as the base env, so placements are
densely graded (the learnable gradient); moves carry only a cost (their value is
purely what they unlock).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from atrium_sim import observations
from atrium_sim.blueprint import (
    Blueprint,
    BrickKind,
    BrickTarget,
    WallSpec,
    brick_face,
    generate_blueprint,
    sample_spec,
)
from atrium_sim.constants import (
    COURSE_MM,
    DROP_ARM_MARGIN_MM,
    FINAL_SETTLE_SUBSTEPS,
    H_MAX,
    L_MAX,
    MAX_SETTLE_SUBSTEPS,
    MOVE_COST_FRAC,
    MOVE_STEP_MM,
    OFFSET_RANGE_MM,
    OVERHANG_MM,
    REACH_MM,
    SPAWN_DROP_MM,
)
from atrium_sim.physics import PhysicsWorld
from atrium_sim.reward import AuditReport, RewardConfig, audit, potential

ROBOT_GLOBALS = 6  # extra features appended to the base observation
OBS_DIM = observations.OBS_DIM + ROBOT_GLOBALS


class Mode(IntEnum):
    PLACE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2


@dataclass(frozen=True)
class RobotEnvConfig:
    reach_mm: float = REACH_MM
    move_step_mm: float = MOVE_STEP_MM
    offset_range_mm: float = OFFSET_RANGE_MM
    move_cost_frac: float = MOVE_COST_FRAC
    invalid_place_frac: float = 0.0   # softened to 0: reach-shaping supplies the move
                                      # incentive, so we no longer punish PLACE-when-stuck
                                      # (that was teaching the policy "placing is bad")
    c_reach: float = 0.5              # potential-based shaping: dense reward for moving the
                                      # base toward the nearest unplaced target (Ng et al.;
                                      # optimal-policy-invariant). Supplies the directional
                                      # movement signal that was missing.
    course_bonus: float = 1.0         # milestone reward for fully completing a course - makes
                                      # "finish this level and move up" a concrete, reachable
                                      # goal (the all-or-nothing completion bonus is too far).
    wander_threshold: int = 3         # a run of >= this many consecutive MOVEs with no brick
                                      # placed in between is "wandering"; each move at/after the
                                      # threshold takes wander_penalty. Directly attacks the
                                      # degenerate "move forever, never place" policy (which is
                                      # exactly how the agent fails on out-of-distribution walls).
    wander_penalty_frac: float = 0.1  # per-move penalty once wandering (x r_scale/n_targets);
                                      # a successful placement resets the streak to 0.
    random_start: bool = True         # start the base at a random point on the rail (not always
                                      # x=0), so work is sometimes to the LEFT and the agent has
                                      # to learn MOVE_LEFT - without this it only ever sweeps right,
                                      # can't backtrack for gaps, and never returns to build up.
    drop_control: bool = False        # when True the model chooses the RELEASE HEIGHT: the arm
                                      # homes at the wall top and box[1] (the otherwise-vestigial
                                      # kind dim) picks how far to lower it before release. Impact
                                      # velocity is then an emergent consequence of the fall.
                                      # False => identical to before (fixed gentle drop).
    arm_margin_mm: float = DROP_ARM_MARGIN_MM   # arm "home" height above the wall top (drop mode)
    drop_penalty_frac: float = 0.0    # drop mode: penalize the release HEIGHT (penalty ~
                                      # fall_frac x this x r_scale/n) so slamming bricks from
                                      # the top costs reward -> pushes toward gentle placement.
    max_settle_substeps: int = MAX_SETTLE_SUBSTEPS
    final_settle_substeps: int = FINAL_SETTLE_SUBSTEPS
    overhang_mm: float = OVERHANG_MM
    suite: str = "train"


class BrickLayerRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, env_cfg=None, reward_cfg=None):
        self.env_cfg = env_cfg or RobotEnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # hybrid action: discrete mode + continuous [offset, kind]
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        ))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        self._renderer = None
        self.frame_sink: list | None = None
        self.tick_callback = None
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
        self._support = self._compute_support(self.blueprint)
        self.world = PhysicsWorld(self.blueprint.length)
        traversal = int(np.ceil(self.blueprint.length / self.env_cfg.move_step_mm))
        # base start: random module-aligned position (forces bidirectional movement)
        if self.env_cfg.random_start:
            self.base_x = min(float(self.np_random.integers(0, traversal + 1))
                              * self.env_cfg.move_step_mm, self.blueprint.length)
        else:
            self.base_x = 0.0
        # step budget = placement budget + a generous move allowance (enough for a
        # per-course sweep on the biggest walls); episodes end on success first
        self.budget = self.blueprint.budget + 2 * self.blueprint.n_courses * traversal + 8
        self.steps = 0
        self.placements = 0
        self.moves = 0
        self._moves_since_place = 0   # consecutive MOVEs since the last brick placed (wander)
        self._release_y = None        # drop-control: last release height (for the renderer)
        self._arm_top_y = None
        self._fall_frac = 0.0         # drop-control: last normalized drop height (for the penalty)
        self.halves_used = 0
        self.off_canvas = 0
        self.invalid = 0
        self.last_disturbance = 0.0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self._step_costs = 0.0
        self._terminal_terms = 0.0
        self._completed_courses = 0
        self.report = self._audit()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), self._info(terminal=False)

    def step(self, action):
        cfg = self.reward_cfg
        n = self.blueprint.n_targets
        prev_report = self.report
        prev_matched = len(prev_report.matches)
        mode_raw, box = action
        mode = Mode(int(mode_raw))
        box = np.asarray(box, dtype=np.float64)

        self.steps += 1
        reward = 0.0
        step_cost = cfg.c_step_frac * cfg.r_scale / n
        self._step_costs += step_cost
        reward -= step_cost

        # potential-based reach shaping: dense reward for moving toward work
        prev_reach = self._reach_potential()
        if mode == Mode.PLACE:
            reward += self._do_place(box)
        else:
            reward += self._do_move(mode)
        reward += self._reach_potential() - prev_reach

        # milestone bonus for each newly-completed course (rewards stacking levels)
        completed = self._completed_course_count()
        if completed > self._completed_courses:
            bonus = self.env_cfg.course_bonus * (completed - self._completed_courses)
            reward += bonus
            self._terminal_terms += bonus
            self._completed_courses = completed

        # termination - all terminal paths final-settle + re-audit first
        terminated = False
        dropped = prev_matched - len(self.report.matches)
        collapse = dropped >= max(cfg.collapse_min_bricks, int(np.ceil(cfg.collapse_frac * prev_matched)))
        complete = not self.report.missing_targets
        budget_out = self.steps >= self.budget

        if collapse or complete or budget_out:
            reward += self._final_settle_delta()
            complete = not self.report.missing_targets
            if collapse:
                reward -= cfg.collapse_penalty
                self._terminal_terms -= cfg.collapse_penalty
                terminated = True
            elif complete:
                bonus = cfg.bonus_fill + (cfg.bonus_perfect if self.report.frac_in_tol == 1.0 else 0.0)
                reward += bonus
                self._terminal_terms += bonus
                terminated = True
            elif budget_out:
                terminated = True

        self.last_reward = float(reward)
        self.episode_return += float(reward)
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), float(reward), terminated, False, self._info(terminal=terminated)

    # --- actions --------------------------------------------------------------

    def _do_place(self, box: np.ndarray) -> float:
        prev_phi = potential(self.report, self.reward_cfg)
        target = self._next_place_target()
        if target is None:  # PLACE with nothing reachable: wasted step
            self.invalid += 1
            return -self.env_cfg.invalid_place_frac * self.reward_cfg.r_scale / self.blueprint.n_targets
        # kind is dictated by the blueprint slot (a masonry robot is TOLD which brick
        # the plan calls for). The agent controls navigation + the placement offset
        # (box[0]); box[1] is the release height ONLY in drop_control mode (else unused).
        kind = target.kind
        if kind == BrickKind.HALF:
            self.halves_used += 1
        w = brick_face(kind)[0]
        x = target.x + float(box[0]) * self.env_cfg.offset_range_mm
        lo = w / 2.0 - self.env_cfg.overhang_mm
        hi = self.blueprint.length - w / 2.0 + self.env_cfg.overhang_mm
        x = float(np.clip(x, lo, hi))
        self.placements += 1
        self._moves_since_place = 0  # placing a brick resets the wander streak
        release_y = self._release_height(target.course, box) if self.env_cfg.drop_control else None
        self._release_y = release_y
        pre = self.world.positions()
        bid = self.world.spawn_brick(x, kind, target.course, release_y=release_y)
        if bid is None:
            self.off_canvas += 1
        removed = self._settle(self.env_cfg.max_settle_substeps)
        self.off_canvas += len(removed)
        self.report = self._audit()
        self.last_disturbance = self._disturbance(pre, bid)
        reward = potential(self.report, self.reward_cfg) - prev_phi
        # penalize a high release (penalty ~ drop height ~ impact energy) so the model
        # is pushed toward realistic gentle placement instead of slamming from the top
        if release_y is not None and self.env_cfg.drop_penalty_frac > 0.0:
            reward -= (self.env_cfg.drop_penalty_frac * self._fall_frac
                       * self.reward_cfg.r_scale / self.blueprint.n_targets)
        return reward

    def _release_height(self, course: int, box: np.ndarray) -> float:
        """Drop-control: box[1] in [-1,1] picks how far to lower the arm from its home
        at the wall top before releasing. box[1]=+1 -> fully lowered (gentle, identical
        to the fixed drop); box[1]=-1 -> released from the top (longest fall, hardest
        impact). Impact velocity is thus emergent from the fall distance, not chosen."""
        lower_frac = (float(box[1]) + 1.0) / 2.0
        gentle_y = COURSE_MM * (course + 0.5) + SPAWN_DROP_MM
        arm_top_y = COURSE_MM * self.blueprint.n_courses + self.env_cfg.arm_margin_mm
        self._arm_top_y = arm_top_y
        release_y = gentle_y + (1.0 - lower_frac) * max(0.0, arm_top_y - gentle_y)
        release_y = float(np.clip(release_y, gentle_y, H_MAX + 120.0 - 1.0))
        # normalized drop height (0 gentle .. 1 released from the top) for the penalty
        span = max(1.0, arm_top_y - gentle_y)
        self._fall_frac = float(np.clip((release_y - gentle_y) / span, 0.0, 1.0))
        return release_y

    def _do_move(self, mode: Mode) -> float:
        step = self.env_cfg.move_step_mm * (-1.0 if mode == Mode.MOVE_LEFT else 1.0)
        self.base_x = float(np.clip(self.base_x + step, 0.0, self.blueprint.length))
        self.moves += 1
        self._moves_since_place += 1
        # audit unchanged by a move, but keep report fresh for obs consistency
        scale = self.reward_cfg.r_scale / self.blueprint.n_targets
        cost = self.env_cfg.move_cost_frac * scale
        # anti-wander: penalize each move once a run of >= wander_threshold consecutive
        # moves has passed with no brick placed (a placement resets the streak)
        if self._moves_since_place >= self.env_cfg.wander_threshold:
            cost += self.env_cfg.wander_penalty_frac * scale
        return -cost

    # --- reachability ---------------------------------------------------------

    @staticmethod
    def _compute_support(bp: Blueprint) -> dict[int, list[int]]:
        """Map each course>0 target to the course-below targets it physically rests
        on (face spans overlapping by >30mm). Precomputed once per reset."""
        support: dict[int, list[int]] = {}
        for c in range(1, bp.n_courses):
            below = bp.course_targets(c - 1)
            for t in bp.course_targets(c):
                tw = brick_face(t.kind)[0]
                ta, tb = t.x - tw / 2, t.x + tw / 2
                support[t.tid] = [
                    b.tid for b in below
                    if min(tb, b.x + brick_face(b.kind)[0] / 2)
                    - max(ta, b.x - brick_face(b.kind)[0] / 2) > 30.0
                ]
        return support

    def _active_course(self) -> int:
        """Lowest course with an unfilled target (obs feature only)."""
        matched = {m.target_id for m in self.report.matches}
        return min((t.course for t in self.blueprint.targets if t.tid not in matched),
                   default=self.blueprint.n_courses - 1)

    def _matched_ids(self) -> set[int]:
        return {m.target_id for m in self.report.matches}

    def _supported(self, t: BrickTarget, matched: set[int]) -> bool:
        """A brick is placeable once the bricks it rests on exist (course 0 always).

        This replaces the old 'complete each course across the whole wall first'
        rule. Support-based placeability lets the robot build a rising left-to-right
        staircase - the left half-brick is laid as soon as its support exists - so
        the wall finishes in ONE rightward sweep with no backtracking (which is what
        blocked every earlier run at ~68%)."""
        return t.course == 0 or all(b in matched for b in self._support[t.tid])

    def _placeable(self, matched: set[int]) -> list[BrickTarget]:
        return [t for t in self.blueprint.targets
                if t.tid not in matched and self._supported(t, matched)]

    def _reachable_open(self) -> list[BrickTarget]:
        matched = self._matched_ids()
        return [t for t in self._placeable(matched)
                if abs(t.x - self.base_x) <= self.env_cfg.reach_mm]

    def _next_place_target(self) -> BrickTarget | None:
        """Leftmost placeable (supported) reachable target -> the staircase order."""
        cand = self._reachable_open()
        return min(cand, key=lambda t: (t.x, t.course)) if cand else None

    def _nearest_open(self) -> BrickTarget | None:
        """Nearest placeable unplaced target (fall back to any) - where to move."""
        matched = self._matched_ids()
        opens = self._placeable(matched) or [
            t for t in self.blueprint.targets if t.tid not in matched
        ]
        return min(opens, key=lambda t: abs(t.x - self.base_x)) if opens else None

    def _completed_course_count(self) -> int:
        """Number of courses with every target matched (fully-built levels)."""
        matched = {m.target_id for m in self.report.matches}
        return sum(
            1 for c in range(self.blueprint.n_courses)
            if all(t.tid in matched for t in self.blueprint.course_targets(c))
        )

    def _reach_potential(self) -> float:
        """Phi_reach(s) = -c_reach * dist(base, nearest unplaced target) / L.

        Moving toward work raises it (positive shaping reward); zero once the wall
        is complete. Potential-based, so it doesn't change the optimal policy -
        it only supplies the directional movement gradient PPO was missing."""
        t = self._nearest_open()
        if t is None:
            return 0.0
        return -self.env_cfg.c_reach * abs(t.x - self.base_x) / self.blueprint.length

    def _min_moves(self) -> int:
        span = self.blueprint.length
        return max(0, int(np.ceil(span / max(1.0, self.env_cfg.reach_mm))))

    # --- internals ------------------------------------------------------------

    def _audit(self) -> AuditReport:
        return audit(self.world.poses(), self.blueprint, self.reward_cfg,
                     off_canvas=self.off_canvas, halves_used=self.halves_used)

    def _settle(self, substeps: int) -> list[int]:
        cbs = []
        if self.render_mode == "human" or self.frame_sink is not None:
            cbs.append(self._render_frame)
        if self.tick_callback is not None:
            cbs.append(self.tick_callback)
        cb = None if not cbs else (cbs[0] if len(cbs) == 1 else (lambda: [c() for c in cbs]))
        _, removed = self.world.settle(substeps, frame_cb=cb)
        return removed

    def _final_settle_delta(self) -> float:
        before = potential(self.report, self.reward_cfg)
        self.off_canvas += len(self._settle(self.env_cfg.final_settle_substeps))
        self.report = self._audit()
        return potential(self.report, self.reward_cfg) - before

    def _disturbance(self, pre, new_id) -> float:
        post = self.world.positions()
        moved = [
            float(np.hypot(post[b][0] - x, post[b][1] - y))
            for b, (x, y) in pre.items() if b in post and b != new_id
        ]
        return max(moved, default=0.0)

    def _obs(self) -> np.ndarray:
        matched_ids = {m.target_id for m in self.report.matches}
        cursor = min((t.course for t in self.blueprint.targets if t.tid not in matched_ids),
                     default=self.blueprint.n_courses - 1)
        course = self.blueprint.course_targets(cursor)
        reachable = self._reachable_open()
        next_t = self._next_place_target()
        g = observations.GlobalState(
            cursor=cursor,
            course_fill_frac=sum(1 for t in course if t.tid in matched_ids) / len(course),
            bricks_left=max(0, self.budget - self.steps),
            budget=self.budget,
            cuts=self.halves_used,
            n_strays=len(self.report.stray_bricks),
            last_disturbance_mm=self.last_disturbance,
            next_slot_x=next_t.x if next_t else 0.0,
            next_slot_is_half=1.0 if (next_t and next_t.kind == BrickKind.HALF) else 0.0,
        )
        base = observations.encode(self.blueprint, self.report, g)
        nearest = self._nearest_open()
        signed_dist = ((nearest.x - self.base_x) / L_MAX) if nearest else 0.0
        robot = np.array([
            self.base_x / L_MAX,
            self.env_cfg.reach_mm / L_MAX,
            len(reachable) / self.blueprint.n_targets,
            float(np.clip(signed_dist, -1.0, 1.0)),
            1.0 if self.base_x <= 1.0 else 0.0,
            1.0 if self.base_x >= self.blueprint.length - 1.0 else 0.0,
        ], dtype=np.float32)
        return np.clip(np.concatenate([base, robot]), -1.0, 1.0)

    def _info(self, terminal: bool) -> dict[str, Any]:
        r = self.report
        info: dict[str, Any] = {
            "frac_in_tol": r.frac_in_tol, "frac_filled": r.frac_filled,
            "moves": self.moves, "placements": self.placements,
        }
        if terminal:
            info["metrics"] = {
                "frac_in_tol": r.frac_in_tol, "frac_filled": r.frac_filled,
                "waste_frac": r.waste_frac, "waste_count": float(r.waste_count),
                "mean_abs_dev_mm": r.mean_abs_dev_mm, "score": r.score,
                "completed": float(r.frac_filled == 1.0),
                "final_potential": potential(r, self.reward_cfg),
                "terminal_terms": self._terminal_terms, "step_costs": self._step_costs,
                "episode_return": self.episode_return,
                "placements": float(self.placements), "moves": float(self.moves),
                "invalid": float(self.invalid), "steps": float(self.steps),
            }
        return info

    # --- rendering ------------------------------------------------------------

    def render(self):
        return self._render_frame() if self.render_mode == "rgb_array" else None

    def _render_frame(self):
        if self._renderer is None:
            from atrium_sim.render.renderer import PygameRenderer

            self._renderer = PygameRenderer(self.blueprint, self.render_mode)
        frame = self._renderer.draw(
            poses=self.world.poses(), report=self.report,
            hud={
                "spec": f"{self.blueprint.spec.n_modules}m x {self.blueprint.spec.n_courses}c",
                "base": f"{self.base_x:.0f}mm", "moves": self.moves,
                "placed": self.placements, "in tol": f"{self.report.frac_in_tol:.0%}",
                "reward": f"{self.last_reward:+.3f}", "return": f"{self.episode_return:+.2f}",
            },
            cursor=0,
            robot=(self.base_x, self.env_cfg.reach_mm,
                   COURSE_MM * self.blueprint.n_courses + self.env_cfg.arm_margin_mm),
        )
        if self.frame_sink is not None and frame is not None:
            self.frame_sink.append(frame)
        return frame

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
