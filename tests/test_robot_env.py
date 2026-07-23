"""Mobile-robot env: API conformance + the oracle solvability tripwire.

If the scripted coverage oracle can't finish walls that REQUIRE moving, the
robot env (reach, movement, support ordering) is broken - not the agent.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import atrium_sim  # noqa: F401
from atrium_sim.blueprint import WallSpec
from atrium_sim.envs.robot_env import OBS_DIM, Mode
from baselines.robot_oracle import RobotOraclePolicy


def test_check_env():
    env = gym.make("atrium_sim/BrickLayerRobot-v0").unwrapped
    check_env(env, skip_render_check=True)
    env.close()


def test_obs_shape_and_bounds():
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) == (544,)
    for _ in range(10):
        obs, *_ = env.step(env.action_space.sample())
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    env.close()


def test_moving_is_required_and_costs():
    """A wall longer than the reach window can't be finished without moving,
    and each move carries a cost (reward strictly negative on a move step)."""
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(random_start=False)  # deterministic base=0
    obs, _ = env.reset(seed=1, options={"spec": WallSpec(8, 5)})
    u = env.unwrapped
    assert u.blueprint.length > u.env_cfg.reach_mm  # unreachable from a fixed base
    _, reward, *_ = env.step((int(Mode.MOVE_RIGHT), np.array([0.0, -0.5], np.float32)))
    assert reward < 0  # a move earns nothing, only costs
    assert u.moves == 1 and u.base_x == u.env_cfg.move_step_mm
    env.close()


@pytest.mark.parametrize("spec", [WallSpec(4, 2), WallSpec(7, 3), WallSpec(10, 6)],
                         ids=lambda s: f"{s.n_modules}x{s.n_courses}")
def test_oracle_completes_by_moving(spec):
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    policy = RobotOraclePolicy(env)
    obs, _ = env.reset(seed=2, options={"spec": spec})
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        done = term or trunc
    m = info["metrics"]
    assert m["frac_filled"] == 1.0
    assert m["frac_in_tol"] >= 0.9
    assert m["moves"] > 0  # it genuinely had to move
    env.close()
