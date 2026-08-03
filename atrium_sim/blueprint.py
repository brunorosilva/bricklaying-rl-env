"""Wall specifications and blueprint (target brick layout) generation.

A blueprint is the ground truth the agent is scored against: one target pose
per brick of a running-bond (halfsteensverband) wall. Even courses are all
full bricks; odd courses start and end with a half brick so the head joints
land mid-brick on the courses below.

This module owns all wall geometry. It is deliberately free of physics and
gymnasium imports so the reward audit (and, later, the GRPO/VLM pipelines)
can consume blueprints standalone.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from atrium_sim.constants import (
    BRICK_FULL_MM,
    BRICK_HALF_MM,
    COURSE_MM,
    MODULE_MM,
    brick_budget,
    wall_length,
)


class BrickKind(IntEnum):
    FULL = 0
    HALF = 1
    VOUSSOIR = 2   # a tapered arch wedge; see atrium_sim.arch. Has no fixed rectangular
                   # face - its geometry is a per-target wedge_verts polygon, not (w, h).


def brick_face(kind: BrickKind) -> tuple[float, float]:
    """Rendered face (w, h) in mm for a brick kind. Not meaningful for VOUSSOIR (which
    carries its own wedge_verts) - callers must branch on kind before calling this."""
    return BRICK_FULL_MM if kind == BrickKind.FULL else BRICK_HALF_MM


@dataclass(frozen=True)
class WallSpec:
    n_modules: int
    n_courses: int


@dataclass(frozen=True)
class BrickTarget:
    tid: int          # global target id (== index into Blueprint.targets)
    course: int       # 0 = bottom (voussoirs: the springing course, shared by the whole ring)
    slot: int         # index within course, left to right (voussoirs: build_order position -
                      # see atrium_sim.arch - NOT left-to-right x, since the ring builds
                      # symmetrically from both springings toward the keystone)
    x: float          # face-centre x, mm
    y: float          # face-centre y, mm (envelope centre; course 0 rests on ground)
    kind: BrickKind
    theta: float = 0.0  # target orientation, radians (0 = flat/level; nonzero for arch voussoirs
                        # that radiate around a rounded opening)
    wedge_verts: tuple[tuple[float, float], ...] | None = None
                        # VOUSSOIR only: local (centroid-relative) polygon verts for the tapered
                        # wedge shape, at theta=0 (physics.spawn_brick rotates by theta). None for
                        # every FULL/HALF target (back-compat) - brick_face()/FULL_VERTS apply.
    arch_id: int | None = None
                        # VOUSSOIR only: which arch this wedge belongs to, for ring-closure and
                        # strike-survival bookkeeping (multiple arches can be in flight at once).


@dataclass(frozen=True)
class Blueprint:
    spec: WallSpec
    length: float                       # wall length L, mm
    targets: tuple[BrickTarget, ...]    # ordered by (course, slot)
    _courses: tuple[tuple[BrickTarget, ...], ...] = field(repr=False, default=())

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def n_halves(self) -> int:
        return sum(1 for t in self.targets if t.kind == BrickKind.HALF)

    @property
    def n_courses(self) -> int:
        return self.spec.n_courses

    @property
    def budget(self) -> int:
        return brick_budget(self.n_targets)

    def course_targets(self, course: int) -> tuple[BrickTarget, ...]:
        return self._courses[course]


def _course_targets(course: int, n_modules: int, length: float) -> list[tuple[float, BrickKind]]:
    """(x, kind) for one course, left to right."""
    if course % 2 == 0:
        return [(105.0 + MODULE_MM * k, BrickKind.FULL) for k in range(n_modules)]
    bricks: list[tuple[float, BrickKind]] = [(50.0, BrickKind.HALF)]
    bricks += [(215.0 + MODULE_MM * k, BrickKind.FULL) for k in range(n_modules - 1)]
    bricks.append((length - 50.0, BrickKind.HALF))
    return bricks


def generate_blueprint(spec: WallSpec) -> Blueprint:
    """Deterministic running-bond blueprint for a wall spec."""
    length = wall_length(spec.n_modules)
    targets: list[BrickTarget] = []
    courses: list[tuple[BrickTarget, ...]] = []
    tid = 0
    for c in range(spec.n_courses):
        y = 30.0 + COURSE_MM * c
        course: list[BrickTarget] = []
        for slot, (x, kind) in enumerate(_course_targets(c, spec.n_modules, length)):
            course.append(BrickTarget(tid=tid, course=c, slot=slot, x=x, y=y, kind=kind))
            tid += 1
        targets.extend(course)
        courses.append(tuple(course))
    return Blueprint(spec=spec, length=length, targets=tuple(targets), _courses=tuple(courses))


def generate_house_blueprint(plan) -> Blueprint:
    """One flat running-bond Blueprint spanning a whole facade grid, in GLOBAL mm.

    A house is NOT N separate panels to the env (which consumes a single Blueprint): reuse
    generate_blueprint per panel, offset its targets to global coords, and re-tid/re-slot by
    (course, x). Opening cells are simply ABSENT (the deterministic tiler never tiled them),
    so under level ordering the robot fills each course around the void. Courses may be ragged
    (or empty, where an opening spans a whole row) - Blueprint._courses already allows that.

    `plan` is any FacadePlan-like object (.panels, .grid_cols, .grid_rows); typed loosely to
    avoid a blueprint<->facade import cycle.

    A real structural arch's springer voussoirs physically OVERSAIL onto the pier/abutment at
    the springing course (real masonry: the ring rests partly ON the pier, per BIA's abutment-
    width rules) - the tiler's panels don't know this (they only carve out the plain
    rectangular opening), so without filtering, an ordinary flat pier brick would be generated
    at the exact same (course, x) the arch ring will occupy, and the two collide physically
    the instant both try to exist there. `plan.arch_regions()` (if present) is consulted to
    drop any flat target inside an arch's actual footprint at its springing course.

    Each panel's running-bond PATTERN (even/odd - full bricks vs. half-ended) is chosen by the
    panel's GLOBAL starting course, not its own local course 0 (which `generate_blueprint`
    always treats as even). Two panels stacked directly on top of each other, tiled
    independently, otherwise get bond parity that's arbitrary relative to each other - normally
    a cosmetic-only mismatch (reported as `bond_violations`, never enforced), but with many
    panels (e.g. an arch's tapered bands) it becomes common enough to matter: a panel whose
    global start happens to be odd, still opening with an all-full "even" course, stacks two
    unstaggered head-joint courses directly on top of each other - discovered in-session as a
    real trigger for cascading topples, not just an aesthetic blemish."""
    raw: list[tuple[int, float, float, BrickKind, float]] = []  # (course, x, y, kind, theta)
    for p in plan.panels:
        length = wall_length(p.spec.n_modules)
        ox = p.origin_col * MODULE_MM
        oy = COURSE_MM * p.origin_row
        for local_c in range(p.spec.n_courses):
            global_c = p.origin_row + local_c
            y = 30.0 + COURSE_MM * local_c
            for x, kind in _course_targets(global_c, p.spec.n_modules, length):
                raw.append((global_c, ox + x, oy + y, kind, 0.0))
    exclusions: list[tuple[int, float, float]] = []
    if hasattr(plan, "arch_regions"):
        for region in plan.arch_regions():
            exclusions.extend(_arch_row_exclusions(region))
    if exclusions:
        # overlap by FULL BRICK WIDTH, not centroid: a brick's edge can foul the ring's
        # footprint by a few mm even when its centroid sits outside the exclusion interval
        # (discovered in-session - a centroid-only check let a brick spawn with its edge
        # physically inside the ring's space, toppling it on settle).
        raw = [
            r for r in raw
            if not any(
                r[0] == c and not (r[1] + brick_face(r[3])[0] / 2.0 <= lo
                                    or r[1] - brick_face(r[3])[0] / 2.0 >= hi)
                for c, lo, hi in exclusions
            )
        ]
    raw.sort(key=lambda r: (r[0], r[1]))
    courses: list[list[BrickTarget]] = [[] for _ in range(plan.grid_rows)]
    targets: list[BrickTarget] = []
    for tid, (course, x, y, kind, theta) in enumerate(raw):
        nt = BrickTarget(tid=tid, course=course, slot=len(courses[course]), x=x, y=y,
                         kind=kind, theta=theta)
        targets.append(nt)
        courses[course].append(nt)
    spec = WallSpec(plan.grid_cols, plan.grid_rows)
    length = plan.grid_cols * MODULE_MM
    return Blueprint(spec=spec, length=length, targets=tuple(targets),
                     _courses=tuple(tuple(c) for c in courses))


def _arch_row_exclusions(region) -> list[tuple[int, float, float]]:
    """[(course, x_lo, x_hi), ...]: every (course, x-range) an arch's ring physically occupies,
    course by course from the springing upward - see atrium_sim.arch.ring_row_spans for why a
    single-row exclusion isn't enough (the ring generally oversails onto the pier across
    several courses near its base, not just the springing course). `region` is any
    ArchRegion-like object (.spec, .origin_x, .springing_course); typed loosely, same reasoning
    as generate_house_blueprint's `plan`."""
    from atrium_sim.arch import ring_row_spans

    out = []
    for row, intervals in ring_row_spans(region.spec).items():
        for lo, hi in intervals:
            out.append((region.springing_course + row, region.origin_x + lo, region.origin_x + hi))
    return out


# --- Spec suites -------------------------------------------------------------
# Held-out suites let eval demonstrate generalisation to wall sizes never
# trained on. INTERP sits inside the training distribution's bounding box;
# EXTRAP sits outside it (larger walls, still within obs padding S_MAX/C_MAX).

INTERP_SPECS: tuple[WallSpec, ...] = (WallSpec(5, 4), WallSpec(7, 3), WallSpec(6, 5))
EXTRAP_SPECS: tuple[WallSpec, ...] = (WallSpec(9, 5), WallSpec(10, 6))
TRAIN_SPECS: tuple[WallSpec, ...] = tuple(
    WallSpec(m, c)
    for m, c in itertools.product(range(4, 9), range(2, 6))
    if WallSpec(m, c) not in INTERP_SPECS
)

# Mobile-robot suites: small walls that still require moving (3+ modules exceed
# the ~500mm reach) but are short-horizon enough that the agent can complete a
# course, advance, and stack levels - the skill it can't discover on big walls.
ROBOT_EVAL_SPECS: tuple[WallSpec, ...] = (WallSpec(4, 3), WallSpec(5, 2))
ROBOT_SPECS: tuple[WallSpec, ...] = tuple(
    WallSpec(m, c)
    for m, c in itertools.product((3, 4, 5), (2, 3))
    if WallSpec(m, c) not in ROBOT_EVAL_SPECS
)

# Mixed small->big curriculum: small walls give an easy completion signal to learn
# on, big walls force real navigation (and exercise the anti-wander penalty). The
# eval set is HELD OUT of training and spans small/mid/big to measure generalization
# across wall sizes - the axis where the small-only "robot" suite doesn't transfer.
ROBOT_BIG_EVAL_SPECS: tuple[WallSpec, ...] = (
    WallSpec(4, 3), WallSpec(6, 4), WallSpec(8, 5),
    WallSpec(9, 5), WallSpec(10, 6),   # harder held-out cases (beyond the 3-8 module range)
)
ROBOT_BIG_SPECS: tuple[WallSpec, ...] = tuple(
    WallSpec(m, c)
    for m, c in itertools.product(range(3, 9), range(2, 6))
    if WallSpec(m, c) not in ROBOT_BIG_EVAL_SPECS
)

# Held-out BIG-wall eval (the Track-A generalization headline): every spec exceeds the L3
# curriculum frontier (10x6) in >=1 axis, so a policy trained capped at L3 sees these
# zero-shot. (Thin 2x40-style piers are a physics-limit probe, scored separately, not here.)
ROBOT_HUGE_EVAL_SPECS: tuple[WallSpec, ...] = (
    WallSpec(12, 8), WallSpec(16, 6), WallSpec(6, 14), WallSpec(20, 10),
)

# Competence-gated SIZE curriculum: (max_modules, max_courses) per rung. sample_spec(level=L)
# draws uniformly from the WHOLE box up to rung L (keeps small walls in the mix -> no
# catastrophic forgetting); the trainer advances L when frontier competence crosses a
# threshold. Reaches facade scale (wide walls + tall piers) at the top rung.
SIZE_LADDER: tuple[tuple[int, int], ...] = (
    (5, 3),    # L0  N<=16  (== the small "robot" suite)
    (6, 4),    # L1
    (8, 5),    # L2
    (10, 6),   # L3
    (12, 8),   # L4
    (16, 10),  # L5
    (20, 14),  # L6  facade scale
)

_SUITES = {
    "train": TRAIN_SPECS, "interp": INTERP_SPECS, "extrap": EXTRAP_SPECS,
    "robot": ROBOT_SPECS, "robot_eval": ROBOT_EVAL_SPECS,
    "robot_big": ROBOT_BIG_SPECS, "robot_big_eval": ROBOT_BIG_EVAL_SPECS,
    "robot_huge_eval": ROBOT_HUGE_EVAL_SPECS,
}


def frontier_specs(level: int) -> tuple[WallSpec, ...]:
    """A few representative walls at curriculum rung `level`: the corner (hardest at this rung),
    a wide-short, and a narrow-tall. Used to measure frontier competence for advancing."""
    max_m, max_c = SIZE_LADDER[min(level, len(SIZE_LADDER) - 1)]
    return (WallSpec(max_m, max_c), WallSpec(max_m, 2), WallSpec(3, max_c))


def sample_spec(rng: np.random.Generator, suite: str = "train",
                level: int | None = None) -> WallSpec:
    """Fixed-suite sampling by default. If `level` is given (curriculum on), sample uniformly
    from the whole (3..max_modules) x (2..max_courses) box up to that ladder rung."""
    if level is not None:
        max_m, max_c = SIZE_LADDER[min(level, len(SIZE_LADDER) - 1)]
        return WallSpec(int(rng.integers(3, max_m + 1)), int(rng.integers(2, max_c + 1)))
    specs = _SUITES[suite]
    return specs[int(rng.integers(len(specs)))]
