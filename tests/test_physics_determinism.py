"""Same seed + same actions -> bit-identical trajectories (same platform/build)."""

import gymnasium as gym
import numpy as np

import atrium_sim  # noqa: F401
from atrium_sim.blueprint import WallSpec


def rollout(seed: int):
    env = gym.make("atrium_sim/BrickLayer-v0")
    obs, _ = env.reset(seed=seed, options={"spec": WallSpec(5, 3)})
    rng = np.random.default_rng(seed)
    rewards, poses = [], None
    done = False
    while not done:
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        obs, r, terminated, truncated, _ = env.step(a)
        rewards.append(r)
        done = terminated or truncated
    poses = env.unwrapped.world.poses()
    env.close()
    return rewards, poses


def test_identical_rollouts():
    r1, p1 = rollout(11)
    r2, p2 = rollout(11)
    assert r1 == r2  # exact float equality, not approx
    assert p1 == p2
