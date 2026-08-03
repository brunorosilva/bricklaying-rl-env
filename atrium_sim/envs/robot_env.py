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

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from atrium_sim.arch import (
    build_order as arch_build_order,
    ring_drift as arch_ring_drift,
    survived as arch_survived,
    voussoir_quality,
    wedge_verts_mass_kg,
)
from atrium_sim.blueprint import (
    Blueprint,
    BrickKind,
    BrickTarget,
    WallSpec,
    brick_face,
    generate_blueprint,
    generate_house_blueprint,
    sample_spec,
)
from atrium_sim.facade import sample_arch_plan
from atrium_sim.constants import (
    COURSE_MM,
    DROP_ARM_MARGIN_MM,
    FINAL_SETTLE_SUBSTEPS,
    H_MAX,
    MATCH_GATE_RAD,
    MAX_SETTLE_SUBSTEPS,
    MOVE_COST_FRAC,
    MOVE_STEP_MM,
    OFFSET_RANGE_MM,
    OVERHANG_MM,
    REACH_MM,
    SPAWN_DROP_MM,
    VOUSSOIR_TILT_RANGE_DEG,
)
from atrium_sim.physics import PhysicsWorld
from atrium_sim.reward import AuditReport, RewardConfig, audit, potential

# The mobile robot observes a compact SENSOR vector (not the blueprint grid): readings that
# are the env's reaction - proximity to the next target, feedback from the last placement,
# rail position/edges, direction to work, progress. Size-agnostic: identical shape for a
# 4x3 wall or a 40-course pier. See _obs().
_ERR_NORM_MM = 30.0   # dx/dy sensor normaliser (full resolution around +-3mm, clips at +-30)
_REACH_SHAPING_CAP_MM = 2000.0   # _reach_potential's normalizer - a fixed, reach-scale distance
                                  # (4x REACH_MM), not wall length, so a move is worth the same
                                  # shaping reward on every wall size (see _reach_potential)
OBS_DIM = 28   # +6 over the pre-stall-fix 22: far-field walking distance, last-PLACE-invalid,
              # consecutive-invalid streak, and the 3-mode action mask (mask_place/left/right -
              # see train.agent.HybridAgent.mask_dim, which reads these as the LAST mask_dim
              # columns rather than a separate info key). Also invalidates every older
              # checkpoint's obs_dim (robot16 already didn't load at 20; this moves the
              # boundary to 22).


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
    invalid_place_frac: float = 0.02  # was softened to 0.0 - reach-shaping alone was meant to
                                      # supply the move incentive, so PLACE-when-stuck went
                                      # unpunished (avoiding teaching "placing is bad"). With
                                      # PLACE now MASKED whenever nothing is reachable (see
                                      # train.agent's mask_dim), this only ever fires for an
                                      # untrained/exploring/mask-ignoring actor - a small,
                                      # ramped (see _do_place) belt-and-braces cost rather than
                                      # the primary anti-stall lever, which the mask now is.
    c_reach: float = 0.5              # potential-based shaping: dense reward for moving the
                                      # base toward the nearest unplaced target (Ng et al.;
                                      # optimal-policy-invariant). Supplies the directional
                                      # movement signal that was missing.
    course_bonus_frac: float = 0.3    # course-completion milestone, as a fraction of r_scale spread
                                      # over ALL courses: total course mass = course_bonus_frac*r_scale
                                      # for ANY wall height (was a FLAT 1.0/course = n_courses total,
                                      # which on a 12-course wall was 12 > r_scale=10 and drowned the
                                      # /N per-brick precision reward - the reward-scale bug that made
                                      # tall walls collapse). Potential-based, so optimal-policy-invariant.
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
    prefill_prob: float = 0.0         # probability an episode STARTS with a random, support-closed
                                      # (physically stable) partial structure already built, at
                                      # exact targets - the robot must COMPLETE a standing wall
                                      # rather than always build from scratch.
    prefill_max_frac: float = 0.7     # cap on the fraction of the wall pre-placed (random 1..this*n)
    fall_off_edge: bool = False       # a real gantry rides a finite rail: if it's already at an
                                      # end and commands a move further off that end, it drives
                                      # off and topples -> the episode ends (charged fall_penalty).
    fall_penalty: float = 1.0         # reward charged for driving off the end of the rail
    max_settle_substeps: int = MAX_SETTLE_SUBSTEPS
    final_settle_substeps: int = FINAL_SETTLE_SUBSTEPS
    overhang_mm: float = OVERHANG_MM
    voussoir_tilt_range_deg: float = VOUSSOIR_TILT_RANGE_DEG   # box[1]'s tilt-nudge range,
                                      # VOUSSOIR placements only (flat bricks: box[1] inert,
                                      # exactly like release-height's existing pattern)
    arch_ring_closure_frac: float = 0.3   # potential-based, mirrors course_bonus_frac: total
                                      # ring-closure reward mass across ALL arches combined is
                                      # arch_ring_closure_frac*r_scale, for any number/size of
                                      # arches (size-invariant)
    arch_survive_bonus: float = 1.0  # terminal-style bonus, paid the instant a ring survives
                                      # its strike (not gated on episode end - it's a real,
                                      # checkable event the moment the centering comes out)
    arch_collapse_penalty: float = 1.0   # symmetric penalty if a ring does NOT survive its strike
    suite: str = "train"
    curriculum: bool = False          # when True, reset() samples the wall size from the
                                      # curriculum frontier (self._curriculum["level"], a mutable
                                      # holder the trainer advances) instead of the fixed suite -
                                      # a competence-gated size schedule (the generalization lever).
    arch_prob_max: float = 0.0        # cap on the fraction of curriculum episodes that build an
                                      # arch-bearing facade (atrium_sim.facade.sample_arch_plan)
                                      # instead of a plain flat WallSpec. 0.0 (default) is
                                      # byte-identical to before this existed - arches only enter
                                      # training when explicitly opted in via this and curriculum.
    arch_prob_per_level: float = 0.05   # arch_prob ramps by this much per curriculum rung
                                      # (capped at arch_prob_max) - larger/harder arch styles
                                      # only become reachable at higher rungs anyway (see
                                      # sample_arch_plan's own grid-size gate), so this mirrors
                                      # that ramp in HOW OFTEN arches appear, not just WHICH ones.
    scenario_mix: float = 0.0        # fraction of episodes drawn from the oracle-gated scenario
                                      # library (atrium_sim.scenarios.sample) instead of the size/
                                      # arch curriculum - isolated skill practice (a known exact
                                      # walking distance, a void wider than reach, a ragged multi-
                                      # opening course, ...) alongside the general curriculum, the
                                      # same "mixed in, never replacing" pattern as arch_prob_*.
                                      # 0.0 (default) is byte-identical to before this existed.
    max_place_attempts: int = 3      # give-up threshold: a target that fails to seat (spawn
                                      # blocked, or placed but out of the match gate) this many
                                      # CONSECUTIVE times is abandoned - any lingering stray
                                      # brick is removed and the target is excluded from future
                                      # candidates. Without this a genuinely unbuildable slot
                                      # (e.g. a physics defect right above an arch) is offered
                                      # forever: confirmed 500+ wasted steps hammering one target.


class BrickLayerRobotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, env_cfg=None, reward_cfg=None):
        self.env_cfg = env_cfg or RobotEnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # hybrid action: discrete mode + continuous [x-offset, tilt-nudge, release-height].
        # box[1] (tilt) only applies to VOUSSOIR placements (real arch voussoirs); box[2]
        # (release height) only applies when drop_control is on - both inert otherwise,
        # exactly like release-height's existing pattern before this change (a dedicated slot,
        # not an overload, but contextually a no-op when not applicable).
        self.action_space = spaces.Tuple((
            spaces.Discrete(3),
            spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
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
        opts = dict(options or {})
        plan = opts.get("plan")
        explicit_spec = opts.get("spec")
        level = None
        if self.env_cfg.curriculum:
            holder = getattr(self, "_curriculum", None)
            level = holder["level"] if holder else 0
        # scenario library draw: only when the CALLER hasn't already pinned a plan/spec (an
        # explicit options= from a test, webviz replay, or a previous draw this same reset -
        # this only ever runs once). Merges the scenario's own options (plan/spec and possibly
        # prefill_tids/base_x) into `opts` so the prefill/base_x reads below pick them up
        # exactly as if the caller had passed them directly.
        if plan is None and explicit_spec is None and self.env_cfg.scenario_mix > 0.0:
            if float(self.np_random.random()) < self.env_cfg.scenario_mix:
                from atrium_sim.scenarios import sample as sample_scenario

                opts.update(sample_scenario(self.np_random))
                plan = opts.get("plan")
                explicit_spec = opts.get("spec")
        if plan is None and explicit_spec is None and self.env_cfg.curriculum:
            # ramping frequency (see RobotEnvConfig.arch_prob_*): a fraction of curriculum
            # episodes build an arch-bearing facade instead of a flat WallSpec, so training
            # sees the arch mechanic mixed in without ever losing the flat-wall skill (only
            # engaged when arch_prob_max > 0 - byte-identical to before otherwise).
            arch_prob = min(self.env_cfg.arch_prob_max, self.env_cfg.arch_prob_per_level * level)
            if arch_prob > 0.0 and float(self.np_random.random()) < arch_prob:
                plan = sample_arch_plan(self.np_random, level)
        if plan is not None:
            # a whole facade (image -> panels + openings) as ONE flat global-mm blueprint;
            # opening cells are absent, so level ordering fills each course around the void
            self.blueprint = generate_house_blueprint(plan)
        else:
            spec: WallSpec = explicit_spec or sample_spec(
                self.np_random, self.env_cfg.suite, level=level
            )
            self.blueprint = generate_blueprint(spec)
        # static hard bodies (lintels/sills/cement) spawned as the build reaches their course
        self._pending_hard = list(plan.hard_bodies()) if plan is not None and hasattr(plan, "hard_bodies") else []
        self._spawned_hard: set[int] = set()
        self._support = self._compute_support(self.blueprint)
        # real structural arches (see atrium_sim.arch/facade.ArchRegion): voussoir BrickTargets
        # live in a SEPARATE pool, not self.blueprint.targets - an arch's ring isn't course-
        # aligned (a semicircular ring's rise spans several courses of height at varying width),
        # so it can't be folded into the flat per-course grid the way ordinary bricks are.
        self._arch_regions = (
            list(plan.arch_regions()) if plan is not None and hasattr(plan, "arch_regions") else []
        )
        self._arch_targets: dict[int, tuple[BrickTarget, ...]] = {}
        tid_cursor = self.blueprint.n_targets
        for region in self._arch_regions:
            vt = region.voussoir_targets(tid_start=tid_cursor)
            self._arch_targets[region.opening_index] = vt
            tid_cursor += len(vt)
            # spandrel packing closing any gap between the ring's true extrados apex and the
            # crown course (the rise is quantised to whole courses at the springing, but the
            # ring's actual outer height rarely lands exactly on a course boundary) - uses the
            # SAME course-triggered spawn-once mechanism as lintels/sills, just appended here
            # since it's ArchRegion-derived rather than Opening-derived.
            packing = region.crown_packing_hard_body()
            if packing is not None:
                self._pending_hard.append(packing)
            # spandrel packing closing the LEFT/RIGHT gap, at EVERY course from the springing
            # to the crown, between the ring's true per-course reach and the tiler's own
            # worst-case void width (see facade.ArchRegion.spandrel_hard_bodies) - the same
            # spawn-once mechanism, just recurring per course instead of only at the top.
            self._pending_hard.extend(region.spandrel_hard_bodies())
        self._arch_state: dict[int, dict] = {
            region.opening_index: {
                "matched": set(), "brick_ids": {}, "before_strike": {},
                "centering_id": None, "abutments_spawned": False,
                "struck": False, "survived": None, "thrust_n": 0.0,
            }
            for region in self._arch_regions
        }
        self.world = PhysicsWorld(self.blueprint.length)
        traversal = int(np.ceil(self.blueprint.length / self.env_cfg.move_step_mm))
        base_x_opt = opts.get("base_x")
        if base_x_opt is not None:
            # explicit start position (atrium_sim.scenarios, e.g. traverse_d needs an EXACT
            # gap in move-steps, which random_start's uniform draw can't guarantee) - overrides
            # random_start for this episode only.
            self.base_x = float(np.clip(base_x_opt, 0.0, self.blueprint.length))
        elif self.env_cfg.random_start:
            # random module-aligned position (forces bidirectional movement)
            self.base_x = min(float(self.np_random.integers(0, traversal + 1))
                              * self.env_cfg.move_step_mm, self.blueprint.length)
        else:
            self.base_x = 0.0
        # step budget = placement budget + a generous move allowance (enough for a
        # per-course sweep on the biggest walls); episodes end on success first. Voussoirs
        # live OUTSIDE self.blueprint (a separate pool - see reset() above), so
        # blueprint.budget alone doesn't account for them; add their count (+ spares) directly.
        n_voussoirs_total = sum(len(vt) for vt in self._arch_targets.values())
        self.budget = (self.blueprint.budget + n_voussoirs_total + 2 * len(self._arch_regions)
                       + 2 * self.blueprint.n_courses * traversal + 8)
        self.steps = 0
        self.placements = 0
        self.moves = 0
        self._moves_since_place = 0   # consecutive MOVEs since the last brick placed (wander)
        self._release_y = None        # drop-control: last release height (for the renderer)
        self._arm_top_y = None
        self._fall_frac = 0.0         # drop-control: last normalized drop height (for the penalty)
        self._fell = False            # drove off the end of the rail this episode
        self._deadlocked = False      # no action can ever make further progress (give-up path)
        self._last_place = None       # (dx, dy, dtheta, in_tol) of the last placed brick, for the
                                      # placement-feedback sensors (None until the first placement)
        self.halves_used = 0
        self.off_canvas = 0
        self.invalid = 0
        self._attempts: dict[int, int] = {}   # tid -> consecutive failed placement attempts
        self._abandoned: set[int] = set()     # tids given up on (see _record_place_attempt)
        self._last_place_invalid = False      # was the LAST PLACE attempted with nothing reachable
        self._invalid_streak = 0              # consecutive invalid-PLACE attempts, broken by any
                                              # MOVE or any PLACE with a real target (see _obs sensors
                                              # last_place_invalid/consecutive_invalid)
        self.last_disturbance = 0.0
        self.episode_return = 0.0
        self.last_reward = 0.0
        self._step_costs = 0.0
        self._terminal_terms = 0.0
        self._completed_courses = 0
        prefill_tids = opts.get("prefill_tids")
        if prefill_tids is not None:
            self._prefill_tids(prefill_tids)
        elif self.env_cfg.prefill_prob > 0.0 and float(self.np_random.random()) < self.env_cfg.prefill_prob:
            self._random_prefill()
        self.report = self._audit()
        self._completed_courses = self._completed_course_count()  # prefilled courses don't re-award
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

        # potential-based course milestone: Phi_course = course_bonus_frac*r_scale*(completed/n_courses),
        # so the TOTAL course-milestone mass is course_bonus_frac*r_scale for ANY wall height (size-
        # invariant). Applied on the delta (telescoping); a negative delta claws the bonus back if a
        # course de-completes on collapse.
        completed = self._completed_course_count()
        if completed != self._completed_courses:
            per_course = self.env_cfg.course_bonus_frac * cfg.r_scale / self.blueprint.n_courses
            bonus = per_course * (completed - self._completed_courses)
            reward += bonus
            self._terminal_terms += bonus
            self._completed_courses = completed

        # spawn any course-triggered hard bodies (lintels/sills/cement) now that the build has
        # reached their trigger course -> the opening's lintel appears "at the exact brick level"
        for idx, hb in enumerate(self._pending_hard):
            if idx not in self._spawned_hard and hb.trigger_course < completed:
                self.world.spawn_static_body(hb.verts_mm, hb.kind, sensor=(hb.kind == "voussoir"))
                self._spawned_hard.add(idx)

        # termination - all terminal paths final-settle + re-audit first
        terminated = False
        if self._fell:
            # drove off the end of the rail -> the gantry topples off its track; episode
            # ends immediately (the fall penalty was already charged in _do_move)
            terminated = True
        else:
            dropped = prev_matched - len(self.report.matches)
            collapse = dropped >= max(cfg.collapse_min_bricks, int(np.ceil(cfg.collapse_frac * prev_matched)))
            complete = not self.report.missing_targets
            budget_out = self.steps >= self.budget
            deadlocked = self._is_deadlocked()

            if collapse or complete or budget_out or deadlocked:
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
                elif budget_out or deadlocked:
                    terminated = True
            self._deadlocked = deadlocked and terminated and not (collapse or complete)

        self.last_reward = float(reward)
        self.episode_return += float(reward)
        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), float(reward), terminated, False, self._info(terminal=terminated)

    # --- actions --------------------------------------------------------------

    def _note_failed_attempt(self, tid: int) -> bool:
        """Increment the consecutive-failure counter for `tid`. True once
        `max_place_attempts` is reached, telling the caller to give up on this target."""
        self._attempts[tid] = self._attempts.get(tid, 0) + 1
        return self._attempts[tid] >= self.env_cfg.max_place_attempts

    def _clear_attempt(self, tid: int) -> None:
        self._attempts.pop(tid, None)

    def _abandon_ring(self, opening_index: int) -> float:
        """A voussoir slot has failed to seat `max_place_attempts` times running. Voussoirs
        must be placed in strict springings-to-keystone order (see `build_order`), so a slot
        that can never seat means the ring can never close - structurally identical to a
        failed strike. Reuse that exact bookkeeping (struck=True, survived=False) rather
        than a separate flag: `_ready_arch_voussoir_candidates` already stops offering a
        struck arch, and `_supported`'s crown gate already requires struck-and-survived, so
        both consequences (no more voussoirs, crown-and-above permanently blocked) fall out
        for free and stay consistent with a real collapsed ring."""
        st = self._arch_state[opening_index]
        st["struck"] = True
        st["survived"] = False
        self._terminal_terms -= self.env_cfg.arch_collapse_penalty
        return -self.env_cfg.arch_collapse_penalty

    def _do_place(self, box: np.ndarray) -> float:
        target = self._next_place_target()
        if target is None:  # PLACE with nothing reachable: wasted step
            self.invalid += 1
            self._last_place_invalid = True
            self._invalid_streak += 1
            # ramped, not flat: a lone invalid PLACE (e.g. right as the last in-reach target
            # got matched a step ago) costs little; a policy that keeps hammering PLACE into
            # nothing pays more each consecutive time. Belt-and-braces once PLACE is masked
            # (see train.agent's mask_dim) - the mask is a soft constraint on a LEARNED
            # policy's logits, not an env invariant, so an untrained/exploring/oracle-free
            # actor can still choose it.
            ramp = 1.0 + min(self._invalid_streak - 1, 8)
            return -(self.env_cfg.invalid_place_frac * ramp * self.reward_cfg.r_scale
                     / self.blueprint.n_targets)
        self._last_place_invalid = False
        self._invalid_streak = 0
        if target.kind == BrickKind.VOUSSOIR:
            return self._do_place_voussoir(target, box)
        prev_phi = potential(self.report, self.reward_cfg)
        # kind is dictated by the blueprint slot (a masonry robot is TOLD which brick
        # the plan calls for). The agent controls navigation + the placement offset
        # (box[0]); box[2] is the release height ONLY in drop_control mode (else unused);
        # box[1] (tilt) is inert here - VOUSSOIR placements only, see _do_place_voussoir.
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
        bid = self.world.spawn_brick(x, kind, target.course, release_y=release_y,
                                     theta=target.theta, rest_y=target.y)
        if bid is None:
            self.off_canvas += 1
        removed = self._settle(self.env_cfg.max_settle_substeps)
        self.off_canvas += len(removed)
        self.report = self._audit()
        # placement-feedback sensor: how the just-placed brick landed (None if it strayed)
        lm = next((m for m in self.report.matches if m.brick_id == bid), None)
        self._last_place = (lm.dx, lm.dy, lm.dtheta, lm.in_tol) if lm else None
        self.last_disturbance = self._disturbance(pre, bid)
        if lm is not None:
            self._clear_attempt(target.tid)
        elif self._note_failed_attempt(target.tid):
            # give up: this target has failed to seat max_place_attempts times running.
            # A lingering stray brick (bid not None, but out of the match gate) would
            # otherwise physically block the spawn probe at this slot forever (confirmed:
            # 500+ wasted steps hammering one target above a since-fixed physics defect).
            self._abandoned.add(target.tid)
            if bid is not None:
                self.world.remove_brick(bid)
                self.off_canvas += 1
                self.report = self._audit()
        reward = potential(self.report, self.reward_cfg) - prev_phi
        # penalize a high release (penalty ~ drop height ~ impact energy) so the model
        # is pushed toward realistic gentle placement instead of slamming from the top
        if release_y is not None and self.env_cfg.drop_penalty_frac > 0.0:
            reward -= (self.env_cfg.drop_penalty_frac * self._fall_frac
                       * self.reward_cfg.r_scale / self.blueprint.n_targets)
        return reward

    def _release_height(self, course: int, box: np.ndarray) -> float:
        """Drop-control: box[2] in [-1,1] picks how far to lower the arm from its home
        at the wall top before releasing. box[2]=+1 -> fully lowered (gentle, identical
        to the fixed drop); box[2]=-1 -> released from the top (longest fall, hardest
        impact). Impact velocity is thus emergent from the fall distance, not chosen."""
        lower_frac = (float(box[2]) + 1.0) / 2.0
        gentle_y = COURSE_MM * (course + 0.5) + SPAWN_DROP_MM
        arm_top_y = COURSE_MM * self.blueprint.n_courses + self.env_cfg.arm_margin_mm
        self._arm_top_y = arm_top_y
        release_y = gentle_y + (1.0 - lower_frac) * max(0.0, arm_top_y - gentle_y)
        release_y = float(np.clip(release_y, gentle_y, H_MAX + 120.0 - 1.0))
        # normalized drop height (0 gentle .. 1 released from the top) for the penalty
        span = max(1.0, arm_top_y - gentle_y)
        self._fall_frac = float(np.clip((release_y - gentle_y) / span, 0.0, 1.0))
        return release_y

    # --- real structural arches -------------------------------------------------------------

    def _arch_region(self, opening_index: int):
        return next(r for r in self._arch_regions if r.opening_index == opening_index)

    def _arch_ready(self, region) -> bool:
        """True once every pier course below the springing is complete (the same
        "finish-and-level before ascending" rule as ordinary courses)."""
        if region.springing_course <= 0:
            return True
        matched = self._matched_ids()
        return all(t.tid in matched for t in self.blueprint.course_targets(region.springing_course - 1))

    def _ready_arch_voussoir_candidates(self) -> list[BrickTarget]:
        """The single NEXT voussoir (in build order) for every arch whose piers are ready and
        whose ring isn't closed yet - one candidate per open arch, enforcing strict
        springings-to-keystone sequencing (the only order validated to survive striking)."""
        out = []
        for region in self._arch_regions:
            st = self._arch_state[region.opening_index]
            if st["struck"] or not self._arch_ready(region):
                continue
            vt = self._arch_targets[region.opening_index]
            next_slot = len(st["matched"])
            if next_slot >= len(vt):
                continue  # ring closed; strike happens synchronously at closure, not here
            out.append(next(t for t in vt if t.slot == next_slot))
        return out

    def _arch_closure_frac(self) -> float:
        """Ring-closure progress across ALL arches combined, in [0, 1] - the potential for the
        ring-closure reward term (size-invariant: total mass is arch_ring_closure_frac*r_scale
        regardless of how many arches or voussoirs there are)."""
        total = sum(len(vt) for vt in self._arch_targets.values())
        if total == 0:
            return 0.0
        done = sum(len(st["matched"]) for st in self._arch_state.values())
        return done / total

    def _do_place_voussoir(self, target: BrickTarget, box: np.ndarray) -> float:
        region = self._arch_region(target.arch_id)
        st = self._arch_state[target.arch_id]
        if not st["abutments_spawned"]:
            # permanent skewback filler blocks (never struck - see arch.abutment_wedge_verts)
            # + the temporary centering (struck once the ring closes, below).
            for hb in region.abutment_hard_bodies():
                self.world.spawn_static_body(hb.verts_mm, hb.kind)
            cent = region.centering_hard_body()
            st["centering_id"] = self.world.spawn_static_body(cent.verts_mm, cent.kind, sensor=False)
            st["abutments_spawned"] = True

        # box[0] = radial-ish x nudge (same convention as flat bricks); box[1] = tilt nudge,
        # LIVE here (VOUSSOIR placement) unlike for flat bricks; box[2] inert (no drop-control
        # for voussoirs - they seat directly against their neighbours on the centering).
        x_off = float(box[0]) * self.env_cfg.offset_range_mm
        dtheta = float(box[1]) * math.radians(self.env_cfg.voussoir_tilt_range_deg)
        mass = wedge_verts_mass_kg(target.wedge_verts)
        self.placements += 1
        self._moves_since_place = 0
        pre = self.world.positions()
        bid = self.world.spawn_brick(
            target.x + x_off, target.kind, target.course,
            theta=target.theta + dtheta, rest_y=target.y,
            wedge_verts=target.wedge_verts, mass_kg=mass,
        )
        if bid is None:
            # nothing spawned, so the world (and self.report) is unchanged - no re-audit needed
            self.off_canvas += 1
            reward = -self.env_cfg.invalid_place_frac * self.reward_cfg.r_scale / self.blueprint.n_targets
            if self._note_failed_attempt(target.tid):
                reward += self._abandon_ring(target.arch_id)
            return reward
        removed = self._settle(self.env_cfg.max_settle_substeps)
        self.off_canvas += len(removed)
        self.last_disturbance = self._disturbance(pre, bid)

        prev_closure = self._arch_closure_frac()
        seated = bid not in removed
        if seated:
            self._clear_attempt(target.tid)
            st["matched"].add(target.tid)
            st["brick_ids"][target.tid] = bid
            poses = {p.brick_id: p for p in self.world.poses()}
            p = poses.get(bid)
            if p is not None:
                st["before_strike"][target.tid] = (p.x, p.y, p.theta)
                d = float(np.hypot(p.x - target.x, p.y - target.y))
                dth = float(p.theta - target.theta)
                self._last_place = (p.x - target.x, p.y - target.y, dth,
                                     voussoir_quality(d, dth) >= 0.999)
        new_closure = self._arch_closure_frac()
        reward = self.env_cfg.arch_ring_closure_frac * self.reward_cfg.r_scale * (new_closure - prev_closure)

        # give-up: the wedge was ejected during settle (knocked/toppled off-canvas) rather
        # than seating - nothing lingers to remove (physics already cleared it), but the
        # SAME slot would otherwise be re-offered forever (voussoirs are strict build-order,
        # so nothing after this slot can ever place either). Ring-failure bookkeeping is the
        # correct model: it genuinely can never close now.
        if not seated and self._note_failed_attempt(target.tid):
            reward += self._abandon_ring(target.arch_id)

        vt = self._arch_targets[target.arch_id]
        if len(st["matched"]) >= len(vt) and not st["struck"]:
            reward += self._strike_arch(target.arch_id)
        # keep self.report fresh (mirrors the flat-brick path in _do_place): before this,
        # a whole ring's worth of exempted/stray bookkeeping only surfaced on the NEXT
        # flat placement's potential delta - a single spurious reward spike well after the
        # arch was actually built, and stale frac_filled/stray_frac/course_fill sensors
        # for the whole ring-building span.
        self.report = self._audit()
        return reward

    def _strike_arch(self, opening_index: int) -> float:
        """The ring is closed: remove the centering (the arch must now stand on its own),
        settle, and check survival. This is a real, checkable structural event, not a proxy -
        exactly the moment a real centering is struck."""
        st = self._arch_state[opening_index]
        before = dict(st["before_strike"])
        if st["centering_id"] is not None:
            self.world.remove_static_body(st["centering_id"])
        # A struck ring needs MORE settle time than an ordinary placement (in-session
        # validation used 1800 substeps, 3x final_settle_substeps's default 600) - short of
        # that, later flat coursework placed directly above the crown lands on a surface that
        # hasn't fully stopped creeping yet, causing repeated placement failures right above
        # the arch.
        self.off_canvas += len(self._settle(3 * self.env_cfg.final_settle_substeps))
        poses = {p.brick_id: p for p in self.world.poses()}
        after = {
            tid: (poses[bid].x, poses[bid].y, poses[bid].theta) if bid in poses else (1e6, 1e6, 0.0)
            for tid, bid in st["brick_ids"].items()
        }
        drift, tilt = arch_ring_drift(before, after)
        ok = arch_survived(drift, tilt)
        st["struck"] = True
        st["survived"] = ok
        springer_bid = next(iter(st["brick_ids"].values()), None)
        if springer_bid is not None and springer_bid in poses:
            st["thrust_n"] = self.world.contact_normal_impulse(springer_bid) / 1000.0
        bonus = self.env_cfg.arch_survive_bonus if ok else -self.env_cfg.arch_collapse_penalty
        self._terminal_terms += bonus
        return bonus

    def _do_move(self, mode: Mode) -> float:
        # a real gantry rides a finite rail: if it's already at an end and commands a
        # move further off that end, it drives off and topples (episode ends).
        at_left = self.base_x <= 1.0
        at_right = self.base_x >= self.blueprint.length - 1.0
        off_edge = (mode == Mode.MOVE_LEFT and at_left) or (mode == Mode.MOVE_RIGHT and at_right)
        if self.env_cfg.fall_off_edge and off_edge:
            self._fell = True
            self.moves += 1
            self._terminal_terms -= self.env_cfg.fall_penalty
            return -self.env_cfg.fall_penalty
        # distance to the nearest open work BEFORE the move (for the anti-wander gate below)
        nt = self._nearest_open()
        d_before = abs(nt.x - self.base_x) if nt is not None else 0.0
        step = self.env_cfg.move_step_mm * (-1.0 if mode == Mode.MOVE_LEFT else 1.0)
        self.base_x = float(np.clip(self.base_x + step, 0.0, self.blueprint.length))
        self.moves += 1
        self._moves_since_place += 1
        self._invalid_streak = 0  # a MOVE breaks a run of consecutive invalid PLACEs
        # audit unchanged by a move, but keep report fresh for obs consistency
        scale = self.reward_cfg.r_scale / self.blueprint.n_targets
        cost = self.env_cfg.move_cost_frac * scale
        # anti-wander: once a run of >= wander_threshold moves has passed with no placement,
        # penalize ONLY moves that increase the distance to work. Level ordering REQUIRES long
        # empty return-traverses (finish a course at the right end, then cross back to the next
        # course's leftmost target); those REDUCE distance to the next target and must not be
        # punished. Pure dithering/oscillation away from work still is.
        if self._moves_since_place >= self.env_cfg.wander_threshold:
            d_after = abs(nt.x - self.base_x) if nt is not None else 0.0
            if d_after > d_before:
                cost += self.env_cfg.wander_penalty_frac * scale
        return -cost

    def _random_prefill(self) -> None:
        """Pre-place a random FLAT-TOPPED partial structure at exact targets, so the robot has
        to COMPLETE an already-standing, LEVEL wall. Because targets are ordered (course, slot),
        the first `count` of them are whole bottom courses plus a left-to-right prefix of the
        current course - i.e. exactly the level build order, and always physically stable."""
        n = self.blueprint.n_targets
        cap = max(1, min(int(self.env_cfg.prefill_max_frac * n), n - 1))
        count = int(self.np_random.integers(1, cap + 1))   # 1..cap -> 0 < placed < n
        self._prefill_tids(t.tid for t in self.blueprint.targets[:count])

    def _prefill_tids(self, tids) -> None:
        """Pre-place an EXPLICIT target set (by tid) at its exact position - the general form
        of _random_prefill's contiguous prefix, used by atrium_sim.scenarios to isolate a
        specific skill (e.g. "everything is done except one far cluster - walk there and
        finish it"). Support-safe for ANY subset as long as every FULL course below the
        highest targeted one is included too: the level gate only requires the course BELOW
        to be complete, not any left-to-right order within a course, so a course can be
        prefilled with a gap in the middle and every brick in it still rests on the (fully
        prefilled) course below, not on its neighbours. Spawned directly (not via _do_place),
        so it doesn't count as an agent placement."""
        tids = set(tids)
        for t in self.blueprint.targets:
            if t.tid in tids:
                self.world.spawn_brick(t.x, t.kind, t.course, theta=t.theta, rest_y=t.y)
        self._settle(self.env_cfg.final_settle_substeps)

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
        """Level (course-by-course) placeability: a course-c brick is placeable only once
        EVERY brick in course c-1 is placed. This enforces real-bricklaying order - finish
        and level each course across the whole wall before ascending - and REPLACES the old
        support-closed staircase (which climbed diagonally at the left edge and left the top
        courses unbuilt on tall walls, the exact OOD-collapse pattern). The regular
        left-to-right/course-by-course decision is also size-invariant, so it extrapolates to
        walls bigger than trained on. _compute_support/self._support are kept for physics/info;
        this gate is independent of them.

        Real arch voussoirs live in a SEPARATE pool (self._arch_targets), so a flat target's
        OWN course never directly contains them - but a flat target at or above an arch's
        crown_course, over that arch's span, is only genuinely supported once the ring beneath
        it has been STRUCK AND SURVIVED, not merely once the (empty, void) flat course below it
        is trivially "complete". Without this check the level gate - which knows nothing about
        arches - would offer crown-course targets while the ring is still mid-build (or, worse,
        never survives), and a policy would burn its whole budget hammering a target with
        nothing yet beneath it (discovered in-session: 100s of wasted attempts at the exact
        crown course, every time, until this gate was added)."""
        if t.course == 0:
            return True
        if not all(b.tid in matched for b in self.blueprint.course_targets(t.course - 1)):
            return False
        for region in self._arch_regions:
            if t.course < region.crown_course:
                continue
            half = region.spec.span_mm / 2.0
            if region.origin_x - half <= t.x <= region.origin_x + half:
                st = self._arch_state[region.opening_index]
                if not (st["struck"] and st["survived"]):
                    return False
        return True

    def _placeable(self, matched: set[int]) -> list[BrickTarget]:
        return [t for t in self.blueprint.targets
                if t.tid not in matched and t.tid not in self._abandoned
                and self._supported(t, matched)]

    def _is_deadlocked(self) -> bool:
        """True if no action can EVER make further progress: work remains, but no flat
        target currently passes the level/arch gate and no voussoir ring is ready either -
        every path to the rest of the wall is permanently blocked (an abandoned target, or
        a ring that failed and can never close/strike - see _abandon_ring). Distinct from
        budget_out: without this, the only way such a state ends is burning the ENTIRE
        remaining step budget on MOVEs/masked-cost PLACEs that can never help (confirmed:
        exactly this pattern, hundreds of wasted steps, before the give-up path existed)."""
        if not self.report.missing_targets:
            return False
        return (not self._placeable(self._matched_ids())
                and not self._ready_arch_voussoir_candidates())

    def _reachable_open(self) -> list[BrickTarget]:
        matched = self._matched_ids()
        flat = [t for t in self._placeable(matched) if abs(t.x - self.base_x) <= self.env_cfg.reach_mm]
        vous = [t for t in self._ready_arch_voussoir_candidates()
                if abs(t.x - self.base_x) <= self.env_cfg.reach_mm]
        return flat + vous

    def _next_place_target(self) -> BrickTarget | None:
        """Reachable placeable target in boustrophedon (snake) order: fill the active course
        left->right on even courses, right->left on odd. Under the level gate all candidates
        share the lowest incomplete course; snaking finishes each course adjacent to the next
        course's start, so the base needs no full empty return-traverse between courses - which
        is what keeps a tall-wall build inside the move budget (a strict L->R order needs ~2x
        the moves per course and runs out on 12+ course walls). Real arch voussoirs are offered
        one at a time per open arch (build-order enforced upstream), sharing the SAME sort key -
        a voussoir's `course` is its arch's springing course, so it naturally interleaves with
        that row's pier bricks by x position."""
        cand = self._reachable_open()
        if not cand:
            return None
        return min(cand, key=lambda t: (t.course, t.x if t.course % 2 == 0 else -t.x))

    def _nearest_open(self) -> BrickTarget | None:
        """Nearest placeable unplaced target (fall back to any) - where to move."""
        matched = self._matched_ids()
        opens = list(self._placeable(matched)) + self._ready_arch_voussoir_candidates()
        if not opens:
            opens = [t for t in self.blueprint.targets
                     if t.tid not in matched and t.tid not in self._abandoned]
        return min(opens, key=lambda t: abs(t.x - self.base_x)) if opens else None

    def _completed_course_count(self) -> int:
        """Number of courses with every target matched (fully-built levels)."""
        matched = {m.target_id for m in self.report.matches}
        return sum(
            1 for c in range(self.blueprint.n_courses)
            if all(t.tid in matched for t in self.blueprint.course_targets(c))
        )

    def _reach_potential(self) -> float:
        """Phi_reach(s) = -c_reach * min(dist(base, nearest unplaced target), cap) / cap,
        cap = _REACH_SHAPING_CAP_MM.

        Moving toward work raises it (positive shaping reward); zero once the wall
        is complete. Potential-based, so it doesn't change the optimal policy -
        it only supplies the directional movement gradient PPO was missing.

        Normalized by a fixed CAP, not wall length L: dividing by L made a single move worth
        c_reach*move_step_mm/L, which SHRINKS as walls grow (confirmed: 0.33 on a 6x4 training
        wall vs 0.125 on a 3520mm facade at c_reach=2.0) - the identical size-dependence bug
        fixed on the observation side (see _obs's next_dx/nearest_dx). A fixed cap makes a
        move worth the same regardless of wall size; the min(...) saturates the potential for
        targets farther than the cap so it stays bounded (telescopes to a finite total)."""
        t = self._nearest_open()
        if t is None:
            return 0.0
        dist = min(abs(t.x - self.base_x), _REACH_SHAPING_CAP_MM)
        return -self.env_cfg.c_reach * dist / _REACH_SHAPING_CAP_MM

    def _min_moves(self) -> int:
        span = self.blueprint.length
        return max(0, int(np.ceil(span / max(1.0, self.env_cfg.reach_mm))))

    # --- internals ------------------------------------------------------------

    def _audit(self) -> AuditReport:
        # exempt seated voussoirs from the stray/waste count: they live outside
        # self.blueprint (a separate, non-course-aligned pool - see reset()) so
        # match_bricks can never match one to anything by construction.
        exempt = frozenset(
            bid for st in self._arch_state.values() for bid in st["brick_ids"].values()
        )
        return audit(self.world.poses(), self.blueprint, self.reward_cfg,
                     off_canvas=self.off_canvas, halves_used=self.halves_used,
                     exempt_brick_ids=exempt)

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
        """A compact SENSOR vector (OBS_DIM scalars) - the env's reaction, not the grid.
        Everything is normalized relative to the CURRENT wall, so the shape and meaning are
        identical for a 4x3 wall or a 40-course pier (size-agnostic).

        Distance-to-work (next_dx / nearest_dx) is normalized by REACH, not wall length - a
        555mm gap must read the same on a 6x4 training wall as on a 3520mm facade. Length-
        normalizing it was the confirmed root cause of the "stops in place" failure: the
        learned MOVE/PLACE threshold sits at a fixed fraction of whatever obs[8] reads, so on
        any wall past ~2200mm a real, out-of-reach gap could still read BELOW that threshold
        and get misread as "close enough to place." reach_mm/length (obs[3]) was meant to be
        the scale cue that fixes this, but it was measurably the LEAST influential of all 22
        inputs (a policy can't learn to rescale a threshold with a rank-22 feature) - reach-
        normalizing the distance itself removes the need for that rescaling entirely."""
        bp = self.blueprint
        length = max(1.0, bp.length)
        reach = self.env_cfg.reach_mm
        r = self.report
        matched_ids = {m.target_id for m in r.matches}
        reachable = self._reachable_open()
        missing = max(1, len(r.missing_targets))
        next_t = self._next_place_target()          # leftmost reachable placeable, or None
        nearest = self._nearest_open()               # nearest unplaced work (any direction), or None

        # active course completion
        cursor = min((t.course for t in bp.targets if t.tid not in matched_ids),
                     default=bp.n_courses - 1)
        course = bp.course_targets(cursor)
        course_fill = sum(1 for t in course if t.tid in matched_ids) / max(1, len(course))

        # `next_*` describe the boustrophedon-next in-reach target; when nothing is in reach
        # (next_t is None), fall back to `nearest` instead of hard-zeroing - the old all-zero
        # vector was indistinguishable from "the target is exactly at the arm" and pinned 4 of
        # 22 fields dead exactly in the state where the agent most needs directional signal.
        fallback = next_t if next_t is not None else nearest
        if fallback is not None:
            next_dx = np.clip((fallback.x - self.base_x) / reach, -1.0, 1.0)
            next_course = fallback.course / max(1, bp.n_courses)
            next_half = 1.0 if fallback.kind == BrickKind.HALF else 0.0
            next_voussoir = 1.0 if fallback.kind == BrickKind.VOUSSOIR else 0.0
        else:
            next_dx = next_course = next_half = next_voussoir = 0.0
        nearest_dx = np.clip((nearest.x - self.base_x) / reach, -1.0, 1.0) if nearest else 0.0
        # magnitude beyond the sign-only saturation above: how many MOVE_STEP_MMs of walking
        # remain once the target clears reach, in action units (so it means the same thing on
        # every wall). Pairs with next_dx/nearest_dx's sign to fully describe a far target -
        # those alone can't tell "just past reach" from "clear across the wall".
        gap_mm = abs(nearest.x - self.base_x) if nearest is not None else 0.0
        moves_to_target = min(max(0.0, gap_mm - reach) / self.env_cfg.move_step_mm / 24.0, 1.0)
        lp = self._last_place  # (dx, dy, dtheta, in_tol) or None
        arch_closure = self._arch_closure_frac()
        at_left = self.base_x <= 1.0
        at_right = self.base_x >= length - 1.0
        # mask_*: which of the 3 modes is a genuine no-op right now, for train.agent's logit
        # mask (see HybridAgent.mask_dim) - PLACE with nothing reachable, or a MOVE that would
        # merely clamp to the same position. A MOVE toward an edge is a real, consequential
        # action (not masked) when fall_off_edge is on - it topples the gantry, it doesn't no-op.
        mask_place = 1.0 if next_t is not None else 0.0
        mask_left = 0.0 if (at_left and not self.env_cfg.fall_off_edge) else 1.0
        mask_right = 0.0 if (at_right and not self.env_cfg.fall_off_edge) else 1.0

        sensors = np.array([
            # --- rail position ---
            self.base_x / length,                                   # where along the rail
            1.0 if at_left else 0.0,                                # at the left end
            1.0 if at_right else 0.0,                               # at the right end
            min(reach / length, 1.0),                               # reach vs wall width
            # --- work sensing ---
            1.0 if reachable else 0.0,                              # is there a target in reach
            next_dx,                                                # next target x, relative to arm
            next_course,                                            # next target height (fraction)
            next_half,                                              # next target is a half brick
            nearest_dx,                                             # direction+dist to nearest work
            min(len(reachable) / missing, 1.0),                    # how much of what's left is in reach
            # --- placement feedback (reaction to the last brick) ---
            np.clip(lp[0] / _ERR_NORM_MM, -1.0, 1.0) if lp else 0.0,
            np.clip(lp[1] / _ERR_NORM_MM, -1.0, 1.0) if lp else 0.0,
            np.clip(lp[2] / MATCH_GATE_RAD, -1.0, 1.0) if lp else 0.0,
            (1.0 if lp[3] else 0.0) if lp else 0.0,                 # last brick within tolerance
            min(self.last_disturbance / _ERR_NORM_MM, 1.0),        # did it disturb neighbors
            # --- progress ---
            r.frac_filled,
            course_fill,
            max(0, self.budget - self.steps) / max(1, self.budget),
            len(r.stray_bricks) / max(1, bp.n_targets),
            min(self._moves_since_place / (2 * max(1, self.env_cfg.wander_threshold)), 1.0),
            # --- arches (0 for every wall/plan with no arch - a flat wall's obs is otherwise
            # identical to before this feature) ---
            next_voussoir,                                          # next target is a voussoir
            arch_closure,                                           # ring-closure across all arches
            # --- far-field + error signal + action mask (added: the stall-fix pass) ---
            moves_to_target,                                        # walking distance left, once out of reach
            1.0 if self._last_place_invalid else 0.0,               # the last PLACE attempt was invalid
            min(self._invalid_streak / 8.0, 1.0),                  # consecutive invalid PLACEs
            mask_place, mask_left, mask_right,                      # which modes are genuine no-ops now
        ], dtype=np.float32)
        return np.clip(sensors, -1.0, 1.0)

    def _info(self, terminal: bool) -> dict[str, Any]:
        r = self.report
        info: dict[str, Any] = {
            "frac_in_tol": r.frac_in_tol, "frac_filled": r.frac_filled,
            "moves": self.moves, "placements": self.placements,
        }
        if terminal:
            n_arches = len(self._arch_regions)
            struck = [st for st in self._arch_state.values() if st["struck"]]
            survived_n = sum(1 for st in struck if st["survived"])
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
                "fell": float(self._fell),
                "deadlocked": float(self._deadlocked),
                "n_arches": float(n_arches),
                "ring_closure": self._arch_closure_frac(),
                "arch_strike_survival": (survived_n / len(struck)) if struck else 1.0,
                "arch_thrust_n": max((st["thrust_n"] for st in struck), default=0.0),
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
            cursor=self._active_course(),  # was hardcoded 0: the renderer's next-slot
            # highlight always evaluated against course 0, never the robot's real active
            # course, on every robot GIF ever rendered.
            robot=(self.base_x, self.env_cfg.reach_mm,
                   COURSE_MM * self.blueprint.n_courses + self.env_cfg.arm_margin_mm),
            hard_bodies=self.world.hard_poses(),
        )
        if self.frame_sink is not None and frame is not None:
            self.frame_sink.append(frame)
        return frame

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
