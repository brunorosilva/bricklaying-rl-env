"""Record an episode (including settle frames - the money shots) to a GIF."""

from __future__ import annotations

import numpy as np


def record_episode(env, policy, path: str, seed: int | None = None, spec=None) -> dict:
    """Roll one episode, capturing every rendered settle frame. Returns terminal metrics."""
    import imageio.v3 as iio

    unwrapped = env.unwrapped
    frames: list[np.ndarray] = []
    unwrapped.frame_sink = frames
    try:
        options = {"spec": spec} if spec is not None else None
        obs, info = env.reset(seed=seed, options=options)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            done = terminated or truncated
        # hold the final wall for a beat
        last = unwrapped._render_frame()
        frames.extend([last] * 12)
    finally:
        unwrapped.frame_sink = None
    iio.imwrite(path, np.stack(frames), duration=33, loop=0)
    return info["metrics"]
