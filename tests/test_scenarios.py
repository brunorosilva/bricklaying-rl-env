"""Prefill scenarios must be physically stable and oracle-completable."""

import gymnasium as gym
import pytest

import atrium_sim  # noqa: F401
from atrium_sim.blueprint import WallSpec
from baselines.oracle import OraclePolicy

SCENARIOS = ["empty", "prefill_base", "almost", "top_gaps"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_oracle_completes_scenario(scenario):
    env = gym.make("atrium_sim/BrickLayer-v0")
    policy = OraclePolicy(env)
    obs, _ = env.reset(seed=1, options={"spec": WallSpec(5, 4), "scenario": scenario})
    u = env.unwrapped
    prefilled = len(u.report.matches)
    if scenario != "empty":
        assert prefilled > 0, "scenario should pre-place bricks"
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        done = term or trunc
    m = info["metrics"]
    assert m["frac_filled"] == 1.0
    assert m["frac_in_tol"] >= 0.9
    env.close()


def test_prefilled_bricks_are_stable():
    """A solid prefill must not topple during setup (all prefilled slots matched)."""
    env = gym.make("atrium_sim/BrickLayer-v0")
    obs, _ = env.reset(seed=2, options={"spec": WallSpec(6, 4), "scenario": "almost"})
    u = env.unwrapped
    # "almost" prefills every course but the top; none should have fallen
    expected = sum(1 for t in u.blueprint.targets if t.course < u.blueprint.n_courses - 1)
    assert len(u.report.matches) == expected
    assert u.report.waste_count == 0
    env.close()
