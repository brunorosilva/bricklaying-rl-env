"""End-to-end training smoke: a tiny PPO run must finish fast with finite
losses and a loadable checkpoint - and the vector-env plumbing must keep
rewards aligned with episodes (the gymnasium 1.x autoreset guard)."""

import time

import gymnasium as gym
import numpy as np
import pytest

import atrium_sim  # noqa: F401
from train.ppo import Args, main


def test_ppo_smoke(tmp_path):
    t0 = time.perf_counter()
    result = main(
        Args(
            total_timesteps=4096,
            num_envs=2,
            num_steps=64,
            eval_interval=8,
            eval_episodes=2,
            gif_every=0,
            run_dir=str(tmp_path),
            seed=1,
        )
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 60, f"smoke run took {elapsed:.1f}s"
    assert result["global_step"] == 4096
    assert result["sps"] > 0
    assert all(np.isfinite(v) for v in result["losses"].values())
    assert result["eval"]  # at least one in-process eval ran

    from train.agent import CheckpointPolicy

    policy = CheckpointPolicy(result["ckpt"])
    env = gym.make("atrium_sim/BrickLayer-v0")
    obs, _ = env.reset(seed=0)
    action = policy.act(obs)
    assert action.shape == (2,) and np.all(np.abs(action) <= 1.0)
    env.close()


def test_same_step_autoreset_reward_alignment():
    """Accumulated per-env rewards must equal the env's own episode_return at
    the exact step final_info reports it. Fails if autoreset mode or the
    final_info mask/key handling drifts (the silent-GAE-corruption trap)."""
    from train.ppo import make_env

    n = 2
    envs = gym.vector.SyncVectorEnv(
        [make_env("train", 30.0, 4.0, False) for _ in range(n)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs, _ = envs.reset(seed=5)
    acc = np.zeros(n)
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(300):
        actions = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
        obs, rewards, term, trunc, infos = envs.step(actions)
        acc += rewards
        if "final_info" in infos:
            fi = infos["final_info"]["metrics"]
            for i in np.flatnonzero(fi["_episode_return"]):
                assert acc[i] == pytest.approx(fi["episode_return"][i], abs=1e-6)
                acc[i] = 0.0
                checked += 1
        if checked >= 4:
            break
    envs.close()
    assert checked >= 4
