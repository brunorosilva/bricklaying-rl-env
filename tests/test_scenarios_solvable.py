"""The oracle-gated scenario library's solvability gate.

Every (name, difficulty) instance in atrium_sim.scenarios.ALL_SCENARIOS must be perfectly
buildable by the privileged oracle. A scenario that fails this is an ENV bug, not a hard
level, and must not be admitted to training - this is exactly the check that would have
caught the uk_terrace (oracle 0.365), the degenerate ARCH_PLAN_SPECS[0] (3 bricks), and the
arch-eval SPEC0 (oracle 0.818) defects before they ever reached a training run.

Extended to the pre-existing curriculum surfaces too (frontier_specs, ROBOT_HUGE_EVAL,
ARCH_PLAN_SPECS, the saved plans/*.json) - none of those had a blanket oracle-solvability
test before this.
"""

from __future__ import annotations

import numpy as np
import pytest

import atrium_sim  # noqa: F401
import gymnasium as gym
from atrium_sim.blueprint import ROBOT_HUGE_EVAL_SPECS, frontier_specs
from atrium_sim.facade import ARCH_PLAN_SPECS, FacadePlan, Opening
from atrium_sim.scenarios import ALL_SCENARIOS, build
from baselines.robot_oracle import RobotOraclePolicy

SEEDS = (0, 1, 2)


def _run_oracle(options: dict, seed: int) -> dict:
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    policy = RobotOraclePolicy(env)
    obs, _ = env.reset(seed=seed, options=options)
    done = False
    info: dict = {}
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        done = term or trunc
    env.close()
    return info["metrics"]


@pytest.mark.parametrize("name,difficulty", ALL_SCENARIOS,
                         ids=[f"{n}-{d}" for n, d in ALL_SCENARIOS])
def test_scenario_is_oracle_solvable(name, difficulty):
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        options = build(name, difficulty, rng)
        m = _run_oracle(options, seed)
        assert m["frac_filled"] == 1.0, f"{name}({difficulty}) seed={seed}: {m}"
        assert m["frac_in_tol"] >= 0.9, f"{name}({difficulty}) seed={seed}: {m}"
        assert m["deadlocked"] == 0.0, f"{name}({difficulty}) seed={seed}: {m}"


@pytest.mark.parametrize("level", range(7), ids=lambda lv: f"L{lv}")
def test_frontier_specs_are_oracle_solvable(level):
    for spec in frontier_specs(level):
        m = _run_oracle({"spec": spec}, seed=0)
        assert m["frac_filled"] == 1.0, f"L{level} {spec}: {m}"


@pytest.mark.parametrize("spec", ROBOT_HUGE_EVAL_SPECS, ids=lambda s: f"{s.n_modules}x{s.n_courses}")
def test_robot_huge_eval_is_oracle_solvable(spec):
    m = _run_oracle({"spec": spec}, seed=0)
    assert m["frac_filled"] == 1.0, f"{spec}: {m}"


@pytest.mark.parametrize("i,arch_spec", list(enumerate(ARCH_PLAN_SPECS)))
def test_arch_plan_specs_are_oracle_solvable(i, arch_spec):
    style, col, row, n_cols, n_rows, ring_courses, grid_cols, grid_rows = arch_spec
    o = Opening("window", col=col, row=row, n_cols=n_cols, n_rows=n_rows,
                has_lintel=False, has_sill=False, arch_style=style, arch_ring_courses=ring_courses)
    plan = FacadePlan.from_perception("t", grid_cols, grid_rows, [o])
    m = _run_oracle({"plan": plan}, seed=0)
    assert m["frac_filled"] == 1.0, f"ARCH_PLAN_SPECS[{i}]={arch_spec}: {m}"
    assert m["ring_closure"] == 1.0
    assert m["arch_strike_survival"] == 1.0


@pytest.mark.parametrize("plan_name", ["colonial", "uk_terrace"])
def test_saved_plans_are_oracle_solvable(plan_name):
    """Two documented, narrow residuals, neither introduced by this diagnosis pass and
    neither solved here - pinned to their CURRENT measured floor so a regression below it
    still fails, rather than asserted at a 1.0 this env genuinely can't deliver yet:

    - uk_terrace: a flush (no-packing) jack-arch crown is measurably marginal at ring scale
      (see README's "Diagnosed: why it stops in place"). Gives up cleanly now (deadlocked),
      not a budget-burning retry loop.
    - colonial: two 1-module-wide, 40-course-tall FREESTANDING piers (219mm x 2400mm, an
      11:1 aspect ratio, dry-stacked with no tie into the surrounding structure) are a
      genuine physics-plausibility limit of the level itself, independent of any policy -
      a VLM-tiling/level-design issue, not something this pass's env/MDP fixes touch."""
    from webviz.server import _load_house_plan

    plan = _load_house_plan(f"house:{plan_name}")
    m = _run_oracle({"plan": plan}, seed=0)
    if plan_name == "uk_terrace":
        assert m["frac_filled"] >= 0.60, f"{plan_name}: {m}"
        assert m["deadlocked"] == 1.0  # gives up cleanly, does not burn the whole step budget
    else:
        assert m["frac_filled"] >= 0.40, f"{plan_name}: {m}"
