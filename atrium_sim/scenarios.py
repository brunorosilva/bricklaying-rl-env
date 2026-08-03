"""Oracle-gated scenario library: hand-designed levels, each isolating ONE skill the diagnosed
failures needed and the size/arch curricula alone don't reliably exercise.

Every generator returns exactly what `BrickLayerRobotEnv.reset(options=...)` already accepts
(`spec` / `plan` / `prefill_tids` / `base_x`) - no env API beyond the two additions this
diagnosis needed (`prefill_tids`: an explicit target set, not just a random prefix;
`base_x`: an exact start position). A scenario is a `(name, difficulty)` pair; `build(name,
difficulty, rng)` returns the reset-options dict.

The gate is `tests/test_scenarios_solvable.py`: every entry in `ALL_SCENARIOS` must be
`frac_filled == 1.0` for the privileged oracle, across several seeds. A scenario that fails
the gate is an ENV bug, not a hard level, and must not be admitted to training - this is
exactly the check that would have caught the `uk_terrace` (oracle 0.365), degenerate
`ARCH_PLAN_SPECS[0]` (3 bricks), and arch-eval SPEC0 (oracle 0.818) defects before they ever
reached a training run.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from atrium_sim.blueprint import WallSpec, generate_blueprint
from atrium_sim.constants import MODULE_MM, MOVE_STEP_MM, REACH_MM
from atrium_sim.facade import FacadePlan, Opening

Generator = Callable[[np.random.Generator, Any], dict]


# --- traverse_d: walk a KNOWN distance (in move-steps), then place -----------------------
# The single most direct isolation of the measured #1 failure: obs[8]/nearest_dx's
# length-normalized dead band (now reach-relative - see robot_env._obs) let the policy
# misread an out-of-reach gap as "close enough to place" on any wall past ~2200mm. Pinning
# the start at the left end and prefilling everything except one target `gap` move-steps
# beyond reach makes the required behavior unambiguous: MOVE_RIGHT exactly `gap` times, place.

def traverse_d(rng: np.random.Generator, gap_steps: int) -> dict:
    n_courses = int(rng.integers(2, 5))
    target_x = REACH_MM + gap_steps * MOVE_STEP_MM
    n_modules = int(target_x / MODULE_MM) + 4  # wide enough to contain the target + margin
    spec = WallSpec(n_modules, n_courses)
    bp = generate_blueprint(spec)
    top_course = bp.n_courses - 1
    far = min(bp.course_targets(top_course), key=lambda t: abs(t.x - target_x))
    prefill = [t.tid for t in bp.targets if t.tid != far.tid]
    return {"spec": spec, "prefill_tids": prefill, "base_x": 0.0}


# --- cross_void_w: a rectangular void wider than reach -----------------------------------
# The user's original hypothesis, tested directly: void_w in MODULES. The reach-crossing
# threshold is void_w >= 3 (660mm > the 500mm one-sided reach); void_w < 3 should already be
# solvable by any competent policy, so the family spans both sides of that threshold.

def cross_void_w(rng: np.random.Generator, void_w: int) -> dict:
    grid_cols = max(int(rng.integers(10, 15)), void_w + 6)
    grid_rows = int(rng.integers(6, 9))
    row = int(rng.integers(2, grid_rows - 2))
    col = (grid_cols - void_w) // 2
    o = Opening("window", col=col, row=row, n_cols=void_w, n_rows=2,
                has_lintel=True, has_sill=False, arch_style="flat")
    plan = FacadePlan.from_perception("scenario:cross_void_w", grid_cols, grid_rows, [o])
    return {"plan": plan}


# --- ragged_course: two edge slivers, wide apart -----------------------------------------
# The uk_terrace courses-2-through-10 pathology in miniature: two openings leave only a
# narrow middle pier and the two wall edges as flat targets for several courses running.

def ragged_course(rng: np.random.Generator, n_modules: int) -> dict:
    grid_cols = max(n_modules, 9)
    grid_rows = int(rng.integers(6, 10))
    mid_pier_w, edge_margin = 2, 1
    each_w = max(1, (grid_cols - 2 * edge_margin - mid_pier_w) // 2)
    row, n_rows = 1, grid_rows - 3
    o1 = Opening("window", col=edge_margin, row=row, n_cols=each_w, n_rows=n_rows,
                has_lintel=True, arch_style="flat")
    o2 = Opening("window", col=edge_margin + each_w + mid_pier_w, row=row,
                n_cols=each_w, n_rows=n_rows, has_lintel=True, arch_style="flat")
    plan = FacadePlan.from_perception("scenario:ragged_course", grid_cols, grid_rows, [o1, o2])
    return {"plan": plan}


# --- long_traverse / tall_thin: pure width vs. pure height, no confound ------------------

def long_traverse(rng: np.random.Generator, n_modules: int) -> dict:
    return {"spec": WallSpec(n_modules, 2)}


def tall_thin(rng: np.random.Generator, n_courses: int) -> dict:
    return {"spec": WallSpec(3, n_courses)}


# --- arch_ring / arch_crown: isolate the ring, then the ring + crown interface -----------
# arch_ring's grid ends exactly at the crown course, so the episode is over the moment the
# ring closes and survives - no flat coursing above it. arch_crown adds a few courses above,
# exercising exactly the crown_packing_hard_body / voussoir-collision-radius interface this
# whole diagnosis centered on.

_ARCH_GEOM: dict[str, tuple[int, int, int, int, int]] = {
    # style -> (col, row, n_cols, n_rows, ring_courses)
    "jack": (1, 1, 1, 1, 1),
    "semicircular": (2, 0, 3, 8, 2),
    "segmental": (2, 0, 3, 8, 2),
}
_ARCH_GRID_COLS = {"jack": 4, "semicircular": 7, "segmental": 7}


def _arch_opening(style: str) -> tuple[Opening, int, int]:
    col, row, n_cols, n_rows, ring_courses = _ARCH_GEOM[style]
    o = Opening("window", col=col, row=row, n_cols=n_cols, n_rows=n_rows,
                has_lintel=False, has_sill=False, arch_style=style, arch_ring_courses=ring_courses)
    crown_row = row + n_rows
    return o, _ARCH_GRID_COLS[style], crown_row


def arch_ring(rng: np.random.Generator, style: str) -> dict:
    o, grid_cols, crown_row = _arch_opening(style)
    plan = FacadePlan.from_perception("scenario:arch_ring", grid_cols, crown_row, [o])
    return {"plan": plan}


def arch_crown(rng: np.random.Generator, style: str) -> dict:
    o, grid_cols, crown_row = _arch_opening(style)
    plan = FacadePlan.from_perception("scenario:arch_crown", grid_cols, crown_row + 3, [o])
    return {"plan": plan}


# --- multi_arch: uk_terrace decomposed --------------------------------------------------
# semicircular/segmental only: a flush (no-packing) JACK crown is measurably marginal once
# MULTIPLE flat courses stack directly on it (confirmed - see arch_crown's "jack" exclusion
# below), and multi_arch always has coursing above the ring. A single jack ring in isolation
# (arch_ring) is fine; this compounding case is the documented residual, not solved here.

def multi_arch(rng: np.random.Generator, n_arches: int) -> dict:
    styles = ["semicircular", "segmental"][:n_arches]
    grid_cols = 7 * len(styles)
    grid_rows = 11
    openings = [
        Opening("window", col=2 + i * 7, row=0, n_cols=3, n_rows=8,
                has_lintel=False, has_sill=False, arch_style=style, arch_ring_courses=2)
        for i, style in enumerate(styles)
    ]
    plan = FacadePlan.from_perception("scenario:multi_arch", grid_cols, grid_rows, openings)
    return {"plan": plan}


# --- resume_partial: arrive at an unfamiliar mid-build state -----------------------------

def resume_partial(rng: np.random.Generator, fill_frac: float) -> dict:
    n_modules = int(rng.integers(6, 13))
    n_courses = int(rng.integers(4, 9))
    spec = WallSpec(n_modules, n_courses)
    bp = generate_blueprint(spec)
    count = max(1, min(int(fill_frac * bp.n_targets), bp.n_targets - 1))
    prefill = [t.tid for t in bp.targets[:count]]
    return {"spec": spec, "prefill_tids": prefill}


# --- registry ------------------------------------------------------------------------------

SCENARIOS: dict[str, tuple[Generator, tuple]] = {
    "traverse_d": (traverse_d, (1, 3, 6, 10, 15, 20)),
    "cross_void_w": (cross_void_w, (1, 2, 3, 4, 6, 8)),
    "ragged_course": (ragged_course, (8, 10, 14, 20)),
    "long_traverse": (long_traverse, (20, 24, 30)),
    "tall_thin": (tall_thin, (14, 20)),
    "arch_ring": (arch_ring, ("jack", "semicircular", "segmental")),
    # "jack" excluded here (unlike arch_ring): a flush (no-packing) jack crown seats one flat
    # course fine, but measurably compounds tilt/deviation once MULTIPLE courses stack on it
    # directly (confirmed - oracle drops to 0.74 fill with 3 courses above vs 1.0 with one).
    # Same documented residual as uk_terrace's jack region - not solved here; see the README.
    "arch_crown": (arch_crown, ("semicircular", "segmental")),
    "multi_arch": (multi_arch, (2,)),  # capped at 2 (semicircular + segmental only - see multi_arch)
    "resume_partial": (resume_partial, (0.2, 0.5, 0.7)),
}

# every (name, difficulty) instance - the solvability gate iterates this exhaustively;
# the trainer's --scenario-mix samples from it uniformly (see train/ppo_robot.py)
ALL_SCENARIOS: tuple[tuple[str, Any], ...] = tuple(
    (name, difficulty) for name, (_, difficulties) in SCENARIOS.items() for difficulty in difficulties
)


def build(name: str, difficulty: Any, rng: np.random.Generator) -> dict:
    """The reset-options dict for one (name, difficulty) scenario instance."""
    generator, _ = SCENARIOS[name]
    return generator(rng, difficulty)


def sample(rng: np.random.Generator) -> dict:
    """A uniformly random scenario instance - what train.ppo_robot's --scenario-mix draws."""
    name, difficulty = ALL_SCENARIOS[int(rng.integers(len(ALL_SCENARIOS)))]
    return build(name, difficulty, rng)
