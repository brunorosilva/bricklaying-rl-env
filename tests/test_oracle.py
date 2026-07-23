"""The oracle tripwire: if perfect play doesn't score ~perfect, the env or
reward is broken - not the agent. CI runs this on every push."""

import time

import gymnasium as gym

import atrium_sim  # noqa: F401
from atrium_sim.blueprint import WallSpec
from baselines.oracle import OraclePolicy
from train.evaluate import evaluate


def test_oracle_beats_tripwire_on_interp_suite():
    env = gym.make("atrium_sim/BrickLayer-v0")
    policy = OraclePolicy(env)
    result = evaluate(env, policy, "interp", episodes=9)
    env.close()
    m = result["metrics"]
    assert m["frac_in_tol"]["mean"] >= 0.9
    assert m["completed"]["mean"] == 1.0
    assert m["waste_frac"]["mean"] == 0.0


def test_oracle_small_wall_fast_and_perfect():
    env = gym.make("atrium_sim/BrickLayer-v0")
    policy = OraclePolicy(env)
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=0, options={"spec": WallSpec(4, 2)})
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(policy.act(obs))
        done = terminated or truncated
    elapsed = time.perf_counter() - t0
    env.close()
    m = info["metrics"]
    assert m["frac_in_tol"] == 1.0
    assert m["completed"] == 1.0
    assert elapsed < 30.0
