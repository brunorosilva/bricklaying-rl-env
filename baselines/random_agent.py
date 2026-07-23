"""Watch a random agent drop bricks - the M1 demo.

    uv run python -m baselines.random_agent --render human --episodes 3

Headless? Use --gif out.gif instead (or SDL_VIDEODRIVER=dummy).
"""

from __future__ import annotations

import argparse

import gymnasium as gym

import atrium_sim  # noqa: F401  (registers the env)
from baselines.policy import RandomPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", choices=["human", "rgb_array", "none"], default="human")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--suite", default="train")
    parser.add_argument("--gif", default=None, help="record the first episode to this GIF path")
    args = parser.parse_args()

    render_mode = None if args.render == "none" else args.render
    env = gym.make("atrium_sim/BrickLayer-v0", render_mode=render_mode)
    env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(suite=args.suite)
    policy = RandomPolicy(env.action_space, args.seed)

    for ep in range(args.episodes):
        if args.gif and ep == 0:
            from atrium_sim.render.recorder import record_episode

            metrics = record_episode(env, policy, args.gif, seed=args.seed)
        else:
            obs, info = env.reset(seed=args.seed + ep)
            done = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(policy.act(obs))
                done = terminated or truncated
            metrics = info["metrics"]
        print(
            f"episode {ep}: return={metrics['episode_return']:+.2f}  "
            f"in-tol={metrics['frac_in_tol']:.0%}  filled={metrics['frac_filled']:.0%}  "
            f"waste={metrics['waste_count']:.0f}  placements={metrics['placements']:.0f}"
        )
    env.close()


if __name__ == "__main__":
    main()
