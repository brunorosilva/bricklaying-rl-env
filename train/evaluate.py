"""Evaluation: run a policy over a held-out spec suite on fixed seeds.

Seeds are enumerated from a fixed base so every checkpoint and baseline sees
IDENTICAL walls - eval numbers are comparable across the whole project.

    uv run python -m train.evaluate --policy oracle --suite interp --episodes 20
    uv run python -m train.evaluate --checkpoint runs/x/ckpt.pt --suite interp
"""

from __future__ import annotations

import argparse
import json
import statistics

import gymnasium as gym

import atrium_sim  # noqa: F401
from atrium_sim.blueprint import _SUITES

BASE_SEED = 10000

METRIC_KEYS = ("episode_return", "frac_in_tol", "frac_filled", "waste_frac",
               "completed", "score", "mean_abs_dev_mm")


def evaluate(env, policy, suite: str, episodes: int) -> dict:
    """Round-robin the suite's specs on fixed seeds; aggregate terminal metrics."""
    specs = _SUITES[suite]
    per_episode: list[dict] = []
    for i in range(episodes):
        spec = specs[i % len(specs)]
        obs, info = env.reset(seed=BASE_SEED + i, options={"spec": spec})
        done = False
        while not done:
            obs, r, terminated, truncated, info = env.step(policy.act(obs))
            done = terminated or truncated
        per_episode.append(info["metrics"])
    summary = {
        k: {
            "mean": statistics.fmean(ep[k] for ep in per_episode),
            "std": statistics.pstdev([ep[k] for ep in per_episode]),
        }
        for k in METRIC_KEYS
    }
    return {"suite": suite, "episodes": episodes, "metrics": summary}


def load_policy(args, env):
    if args.checkpoint:
        from train.agent import CheckpointPolicy

        return CheckpointPolicy(args.checkpoint)
    from baselines.policy import make_policy

    return make_policy(args.policy, env, seed=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="random", help="oracle | greedy | random")
    parser.add_argument("--checkpoint", default=None, help="trained agent checkpoint (.pt)")
    parser.add_argument("--suite", default="interp", choices=list(_SUITES))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gif", default=None, help="also record one episode to this path")
    args = parser.parse_args()

    env = gym.make("atrium_sim/BrickLayer-v0",
                   render_mode="rgb_array" if args.gif else None)
    policy = load_policy(args, env)
    result = evaluate(env, policy, args.suite, args.episodes)
    result["policy"] = args.checkpoint or args.policy
    if args.gif:
        from atrium_sim.render.recorder import record_episode

        record_episode(env, policy, args.gif, seed=BASE_SEED)
        result["gif"] = args.gif
    env.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
