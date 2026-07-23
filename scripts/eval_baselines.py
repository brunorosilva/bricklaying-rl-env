"""Evaluate all baselines on a suite and write the anchor JSON for plot_curves.

    uv run python scripts/eval_baselines.py --suite interp --episodes 30 --out media/baselines.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym

import atrium_sim  # noqa: F401
from baselines.policy import make_policy
from train.evaluate import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="interp")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("media/baselines.json"))
    args = parser.parse_args()

    anchors = {}
    for name in ("oracle", "greedy", "random"):
        env = gym.make("atrium_sim/BrickLayer-v0")
        result = evaluate(env, make_policy(name, env, seed=0), args.suite, args.episodes)
        env.close()
        anchors[name] = {k: v["mean"] for k, v in result["metrics"].items()}
        print(f"{name}: return={anchors[name]['episode_return']:+.2f} "
              f"in-tol={anchors[name]['frac_in_tol']:.2%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(anchors, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
