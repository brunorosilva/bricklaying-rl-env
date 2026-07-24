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
    assert obs.shape == (OBS_DIM,)
    for _ in range(10):
        obs, *_ = env.step(env.action_space.sample())
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    env.close()


def test_sensor_obs_is_compact_and_size_agnostic():
    """The robot observes a small fixed SENSOR vector (not the blueprint grid), so its
    shape is identical for a tiny wall and a 40-course facade pier."""
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    assert OBS_DIM < 64  # compact sensor vector, not a C_MAX*S_MAX grid
    shapes = set()
    for spec in (WallSpec(4, 3), WallSpec(2, 40), WallSpec(17, 8)):
        obs, _ = env.reset(seed=1, options={"spec": spec})
        assert obs.shape == (OBS_DIM,)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        shapes.add(obs.shape)
    assert len(shapes) == 1  # size-agnostic
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


# --- drop-control (model-chosen release height) -----------------------------

def _run_place_only(box1: float, drop_control: bool):
    """Build a 2x2 wall (fully within reach from base 0) with PLACE-only actions,
    box[1]=box1. Returns (total_reward, frac_filled, settled poses) for comparison."""
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(
        random_start=False, drop_control=drop_control)
    env.reset(seed=3, options={"spec": WallSpec(2, 2)})
    u = env.unwrapped
    total, done = 0.0, False
    while not done:
        _, r, term, trunc, info = env.step((int(Mode.PLACE), np.array([0.0, box1], np.float32)))
        total += r
        done = term or trunc
    poses = tuple((round(p.x, 3), round(p.y, 3)) for p in u.world.poses())
    env.close()
    return round(total, 6), info["metrics"]["frac_filled"], poses


def test_drop_control_off_box1_inert():
    """With drop_control off, box[1] must have zero effect (it's the vestigial kind dim)."""
    assert _run_place_only(1.0, drop_control=False) == _run_place_only(-1.0, drop_control=False)


def test_drop_gentle_reproduces_fixed_drop():
    """drop_control on with box[1]=+1 (arm fully lowered) == the fixed gentle drop."""
    assert _run_place_only(1.0, drop_control=False) == _run_place_only(1.0, drop_control=True)


def test_drop_control_shapes_unchanged():
    """Enabling drop_control must not change the action or observation shapes."""
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(drop_control=True)
    obs, _ = env.reset(seed=1)
    assert env.action_space.spaces[1].shape == (2,)
    assert obs.shape == (OBS_DIM,)
    env.close()


def test_prefill_off_by_default():
    """Default (prefill_prob=0): the wall starts empty."""
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    env.reset(seed=0, options={"spec": WallSpec(4, 3)})
    assert len(env.unwrapped.report.matches) == 0
    env.close()


def test_prefill_is_stable_and_completable():
    """A prefilled episode starts with a random support-closed partial structure (some
    but not all bricks) that the oracle can finish - i.e. prefill yields valid walls."""
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(prefill_prob=1.0, prefill_max_frac=0.6)
    u = env.unwrapped
    env.reset(seed=0, options={"spec": WallSpec(6, 4)})
    pre = len(u.report.matches)
    assert 0 < pre < u.blueprint.n_targets  # some, not all, pre-placed
    policy = RobotOraclePolicy(env)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(policy.act(u._obs()))
        done = term or trunc
    assert info["metrics"]["frac_filled"] == 1.0  # oracle completes the standing wall
    env.close()


def test_fall_off_edge():
    """fall_off_edge on: commanding a move further off an edge topples the gantry
    (terminates). Off (default): the move just clamps and the episode continues."""
    mv_left = (int(Mode.MOVE_LEFT), np.array([0.0, 0.0], np.float32))
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(fall_off_edge=True, random_start=False)
    env.reset(seed=1, options={"spec": WallSpec(4, 3)})
    _, _, term, _, info = env.step(mv_left)  # at base=0, move further left -> fall
    assert term and env.unwrapped._fell and info["metrics"]["fell"] == 1.0
    env.close()

    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(fall_off_edge=False, random_start=False)
    env.reset(seed=1, options={"spec": WallSpec(4, 3)})
    _, _, term, _, _ = env.step(mv_left)
    assert not term and not env.unwrapped._fell and env.unwrapped.base_x == 0.0
    env.close()


def test_release_height_monotone_and_endpoints():
    """box[1]=+1 -> gentle height; box[1]=-1 -> arm top; strictly monotone between."""
    from atrium_sim.constants import COURSE_MM, SPAWN_DROP_MM

    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    u = env.unwrapped
    u.env_cfg = type(u.env_cfg)(drop_control=True)
    env.reset(seed=1, options={"spec": WallSpec(4, 3)})
    gentle = COURSE_MM * 0.5 + SPAWN_DROP_MM
    top = COURSE_MM * u.blueprint.n_courses + u.env_cfg.arm_margin_mm
    rh = lambda b1: u._release_height(0, np.array([0.0, b1], np.float32))
    assert rh(1.0) == pytest.approx(gentle)
    assert rh(-1.0) == pytest.approx(top)
    assert rh(1.0) < rh(0.0) < rh(-1.0)
    env.close()
