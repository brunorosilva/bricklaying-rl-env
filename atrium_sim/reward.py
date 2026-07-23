"""The reward IS the audit — atrium-sim's centrepiece.

A single pure function `audit(bricks, blueprint, cfg)` scores a settled wall
against its blueprint, exactly like a site QA inspection:

- each blueprint target is matched to at most one placed brick (same kind,
  within a 55mm gate);
- a matched brick earns quality q = s_pos(d) * s_ang(theta): full credit on
  the BIM tolerance plateau (±3mm, ±0.5°), smooth Gaussian decay outside;
- unmatched bricks are strays; strays + off-canvas bricks + unnecessary
  half-brick cuts are waste.

The audit defines a potential Phi(wall). The environment's per-step reward is
the *change* in potential (potential-based shaping, Ng et al. 1999), so:

- toppling a brick three steps after placing it is clawed back automatically
  (Phi drops when it happens - no event bookkeeping);
- the undiscounted episode return telescopes to the final audit score, the
  same number published in eval tables and reused later as the GRPO
  sequence-level reward.

Everything here is deterministic, geometry-only, and has no access to the
simulator: `BrickPose` tuples in, `AuditReport` out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from atrium_sim.blueprint import Blueprint
from atrium_sim.constants import MATCH_GATE_MM, MATCH_GATE_RAD, TOL_DEG, TOL_MM
from atrium_sim.physics import BrickPose


@dataclass(frozen=True)
class RewardConfig:
    # --- tolerance & shaping ---
    tol_mm: float = TOL_MM          # BIM position tolerance; full-reward plateau half-width
    sigma_mm: float = 12.0          # Gaussian shoulder beyond the plateau; curriculum: 25 -> 12
    tol_deg: float = TOL_DEG        # levelness plateau half-width
    sigma_deg: float = 2.0          # angular shoulder; curriculum: 4 -> 2
    match_gate_mm: float = MATCH_GATE_MM  # MUST stay < 60 = half the min same-kind target distance
    match_gate_rad: float = MATCH_GATE_RAD
    # --- aggregation ---
    r_scale: float = 10.0           # accuracy mass of a fully perfect wall
    c_waste: float = 0.5            # cost of one wasted brick, in perfect-brick equivalents
    c_step_frac: float = 0.02       # per-step cost as a fraction of one perfect brick's reward
    bonus_fill: float = 1.0         # terminal: all targets matched (within gate)
    bonus_perfect: float = 2.0      # terminal: frac_in_tol == 1.0
    collapse_penalty: float = 2.0   # terminal extra on collapse (on top of the natural dPhi crater)
    collapse_min_bricks: int = 3    # collapse = matched count drops by >= max(this, frac*prev)
    collapse_frac: float = 0.25
    # --- reporting only (no reward effect) ---
    bond_align_mm: float = 45.0     # adjacent-course head joints closer than this = bond violation

    @property
    def step_cost(self) -> float:
        """Per-step cost per blueprint target: c_step_frac * r_scale / N (N applied by caller)."""
        return self.c_step_frac * self.r_scale


@dataclass(frozen=True)
class Match:
    brick_id: int
    target_id: int
    dx: float           # mm, brick - target
    dy: float           # mm
    dtheta: float       # radians, folded to [-pi/2, pi/2)
    d: float            # Euclidean position error, mm
    q: float            # quality in [0, 1]
    in_tol: bool        # d <= tol_mm and |dtheta| <= tol_deg


@dataclass(frozen=True)
class AuditReport:
    matches: tuple[Match, ...]
    missing_targets: tuple[int, ...]   # target ids with no matched brick
    stray_bricks: tuple[int, ...]      # placed brick ids matched to no target
    n_targets: int
    frac_filled: float                 # len(matches) / n_targets
    frac_in_tol: float                 # HEADLINE metric: in-tol matches / n_targets
    mean_abs_dev_mm: float             # over matches (0 if none)
    p95_abs_dev_mm: float
    waste_count: int                   # strays + off-canvas + unnecessary cuts
    waste_frac: float
    bond_violations: int               # reported only, never rewarded
    course_level_dev_mm: tuple[float, ...]  # per-course mean |dy|, reported only
    plumb_dev_mm: float                # max |dx| among course-end matches, reported only
    score: float                       # Phi / r_scale in (-inf, 1]: the normalised wall score


def fold_angle(theta: float) -> float:
    """Fold an unbounded body angle to [-pi/2, pi/2).

    Bricks are 180°-symmetric: a brick that tumbled and landed flat-but-flipped
    (angle ~ pi) is geometrically perfect and must score as such.
    """
    return (theta + math.pi / 2.0) % math.pi - math.pi / 2.0


def plateau_gauss(v: float, tol: float, sigma: float) -> float:
    """1.0 on the tolerance plateau, C1-continuous Gaussian decay outside."""
    if v <= tol:
        return 1.0
    return math.exp(-(((v - tol) / sigma) ** 2))


def brick_quality(d: float, dtheta: float, cfg: RewardConfig) -> float:
    """q = s_pos * s_ang - multiplicative, so a tilted brick can't farm position reward."""
    s_pos = plateau_gauss(d, cfg.tol_mm, cfg.sigma_mm)
    s_ang = plateau_gauss(abs(math.degrees(dtheta)), cfg.tol_deg, cfg.sigma_deg)
    return s_pos * s_ang


def match_bricks(
    bricks: Sequence[BrickPose], blueprint: Blueprint, cfg: RewardConfig
) -> tuple[list[tuple[BrickPose, int, float, float, float, float]], list[int], list[int]]:
    """Match placed bricks to blueprint targets. Returns (matches, missing, strays).

    Gate: same kind AND Euclidean d <= match_gate_mm AND |folded theta| <= gate.
    Kind is filtered FIRST: the disjointness invariant that makes greedy
    per-target matching optimal is that same-KIND targets are >= 120mm apart
    (> 2 * 55mm gate), so each brick is in-gate for at most one target.
    (Cross-kind targets can be as close as ~81mm - without the kind filter the
    invariant would be false.) Ties broken by lower brick id; each brick is
    consumed by its first match.

    Match tuples: (brick, target_id, dx, dy, dtheta_folded, d).
    """
    consumed: set[int] = set()
    matches: list[tuple[BrickPose, int, float, float, float, float]] = []
    missing: list[int] = []
    for t in blueprint.targets:
        best: tuple[float, int] | None = None  # (d, brick_id) - natural tie-break
        best_data: tuple[BrickPose, float, float, float, float] | None = None
        for b in bricks:
            if b.kind != t.kind or b.brick_id in consumed:
                continue
            theta = fold_angle(b.theta)
            if abs(theta) > cfg.match_gate_rad:
                continue  # a toppled brick must not match-and-block the slot
            dx, dy = b.x - t.x, b.y - t.y
            d = math.hypot(dx, dy)
            if d > cfg.match_gate_mm:
                continue
            if best is None or (d, b.brick_id) < best:
                best = (d, b.brick_id)
                best_data = (b, dx, dy, theta, d)
        if best_data is None:
            missing.append(t.tid)
        else:
            b, dx, dy, theta, d = best_data
            consumed.add(b.brick_id)
            matches.append((b, t.tid, dx, dy, theta, d))
    strays = [b.brick_id for b in bricks if b.brick_id not in consumed]
    return matches, missing, strays


def _bond_violations(
    matches: list[tuple[BrickPose, int, float, float, float, float]],
    blueprint: Blueprint,
    cfg: RewardConfig,
) -> int:
    """Head joints in adjacent courses vertically aligned within bond_align_mm.

    Reported only: bond correctness is already encoded in the target
    coordinates, so rewarding it again would double-count.
    """
    from atrium_sim.blueprint import brick_face

    joints_by_course: dict[int, list[float]] = {}
    by_course: dict[int, list[tuple[float, float]]] = {}
    target_course = {t.tid: t.course for t in blueprint.targets}
    for b, tid, *_ in matches:
        w = brick_face(b.kind)[0]
        by_course.setdefault(target_course[tid], []).append((b.x - w / 2, b.x + w / 2))
    for c, spans in by_course.items():
        spans.sort()
        joints_by_course[c] = [
            (right_edge + next_left) / 2.0
            for (_, right_edge), (next_left, _) in zip(spans, spans[1:])
        ]
    violations = 0
    for c in sorted(joints_by_course):
        for j1 in joints_by_course.get(c, ()):
            for j2 in joints_by_course.get(c + 1, ()):
                if abs(j1 - j2) < cfg.bond_align_mm:
                    violations += 1
    return violations


def audit(
    bricks: Sequence[BrickPose],
    blueprint: Blueprint,
    cfg: RewardConfig,
    *,
    off_canvas: int = 0,
    halves_used: int = 0,
) -> AuditReport:
    """Score a settled wall. Pure geometry: no sim access, no RNG, no learning.

    `off_canvas` (bricks that fell off and were removed, plus skipped spawns)
    and `halves_used` (total half-brick placements attempted) are episode
    counters owned by the environment.
    """
    raw_matches, missing, strays = match_bricks(bricks, blueprint, cfg)
    n = blueprint.n_targets

    matches = tuple(
        Match(
            brick_id=b.brick_id,
            target_id=tid,
            dx=dx,
            dy=dy,
            dtheta=theta,
            d=d,
            q=brick_quality(d, theta, cfg),
            in_tol=d <= cfg.tol_mm and abs(math.degrees(theta)) <= cfg.tol_deg,
        )
        for b, tid, dx, dy, theta, d in raw_matches
    )

    devs = sorted(m.d for m in matches)
    mean_dev = sum(devs) / len(devs) if devs else 0.0
    p95_dev = devs[min(len(devs) - 1, math.ceil(0.95 * len(devs)) - 1)] if devs else 0.0

    waste = len(strays) + off_canvas + max(0, halves_used - blueprint.n_halves)

    # reported-only aggregates
    target_by_id = {t.tid: t for t in blueprint.targets}
    course_dy: dict[int, list[float]] = {}
    end_dx: list[float] = []
    for m in matches:
        t = target_by_id[m.target_id]
        course_dy.setdefault(t.course, []).append(abs(m.dy))
        if t.slot in (0, len(blueprint.course_targets(t.course)) - 1):
            end_dx.append(abs(m.dx))
    course_level = tuple(
        sum(course_dy.get(c, [0.0])) / max(1, len(course_dy.get(c, [])))
        for c in range(blueprint.n_courses)
    )

    sum_q = sum(m.q for m in matches)
    score = (sum_q - cfg.c_waste * waste) / n

    return AuditReport(
        matches=matches,
        missing_targets=tuple(missing),
        stray_bricks=tuple(strays),
        n_targets=n,
        frac_filled=len(matches) / n,
        frac_in_tol=sum(1 for m in matches if m.in_tol) / n,
        mean_abs_dev_mm=mean_dev,
        p95_abs_dev_mm=p95_dev,
        waste_count=waste,
        waste_frac=waste / n,
        bond_violations=_bond_violations(raw_matches, blueprint, cfg),
        course_level_dev_mm=course_level,
        plumb_dev_mm=max(end_dx, default=0.0),
        score=score,
    )


def potential(report: AuditReport, cfg: RewardConfig) -> float:
    """Phi(wall) = r_scale * [sum(q) - c_waste * W] / N. Phi(empty wall) = 0."""
    return cfg.r_scale * report.score
