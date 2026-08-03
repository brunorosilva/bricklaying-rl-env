"""End-to-end training smoke for the ROBOT trainer (train/ppo_robot.py) - a tiny run must
finish fast with finite losses and a loadable, runnable checkpoint. Exercises the full
robot18 config (curriculum + arches + scenario_mix + action_mask) together at a tiny scale,
so a wiring mistake among any of them (argument order, a missing default, an eval function
crashing on a scenario it's never seen) is caught cheaply, before a real multi-hour run."""

import time

import gymnasium as gym
import numpy as np

import atrium_sim  # noqa: F401
from train.ppo_robot import Args, main


def test_ppo_robot_smoke(tmp_path):
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
            curriculum=True,
            curriculum_cap=2,
            arch_prob_max=0.3,
            scenario_mix=0.5,
            action_mask=True,
            random_start=True,
        )
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 90, f"smoke run took {elapsed:.1f}s"
    assert result["global_step"] == 4096
    assert result["sps"] > 0
    assert all(np.isfinite(v) for v in result["losses"].values())
    assert result["eval"]  # at least one in-process eval ran
    assert result["eval_scenarios"] is not None and np.isfinite(result["eval_scenarios"]["mean"])

    from train.agent import load_hybrid_agent, HybridAgentPolicy

    agent = load_hybrid_agent(result["ckpt"])
    assert agent.mask_dim == 3  # action_mask=True persisted into the checkpoint
    policy = HybridAgentPolicy(agent)
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    obs, _ = env.reset(seed=0)
    mode, box = policy.act(obs)
    assert mode in (0, 1, 2) and box.shape == (3,) and np.all(np.abs(box) <= 1.0)
    env.close()


def test_ppo_robot_smoke_masking_disabled(tmp_path):
    """action_mask=False must still train end-to-end (an A/B control against the masked
    default) and persist mask_dim=0 so the checkpoint's own logits are never masked."""
    result = main(
        Args(
            total_timesteps=2048,
            num_envs=2,
            num_steps=64,
            eval_interval=0,
            gif_every=0,
            run_dir=str(tmp_path),
            seed=2,
            action_mask=False,
        )
    )
    assert result["global_step"] == 2048

    from train.agent import load_hybrid_agent

    agent = load_hybrid_agent(result["ckpt"])
    assert agent.mask_dim == 0
