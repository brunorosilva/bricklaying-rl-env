"""Gymnasium API conformance and info picklability."""

import pickle

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import atrium_sim  # noqa: F401
from atrium_sim.observations import OBS_DIM


def test_check_env():
    env = gym.make("atrium_sim/BrickLayer-v0").unwrapped
    check_env(env, skip_render_check=True)
    env.close()


def test_obs_shape_and_bounds():
    env = gym.make("atrium_sim/BrickLayer-v0")
    obs, _ = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,) == (538,)
    for _ in range(5):
        obs, *_ = env.step(env.action_space.sample())
        assert obs.dtype == np.float32
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    env.close()


def test_info_is_picklable():
    env = gym.make("atrium_sim/BrickLayer-v0")
    obs, info = env.reset(seed=4)
    pickle.dumps(info)
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    pickle.dumps(info)
    assert isinstance(info["metrics"], dict)
    assert all(isinstance(v, float) for v in info["metrics"].values())
    env.close()


def test_seeding_reproducible():
    def rollout(seed):
        env = gym.make("atrium_sim/BrickLayer-v0")
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        rs = []
        for _ in range(8):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            rs.append(r)
            if term or trunc:
                break
        env.close()
        return obs, rs

    o1, r1 = rollout(7)
    o2, r2 = rollout(7)
    assert np.array_equal(o1, o2)
    assert r1 == r2
