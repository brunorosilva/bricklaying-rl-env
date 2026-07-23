"""Policy protocol shared by baselines, evaluation and trained agents."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Map one observation to one action."""
        ...


class RandomPolicy:
    """Lower-bound anchor. Doubles as the M1 watchable demo."""

    def __init__(self, action_space, seed: int | None = None):
        self.action_space = action_space
        if seed is not None:
            self.action_space.seed(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


def make_policy(name: str, env, seed: int | None = None) -> Policy:
    """Baseline factory used by evaluate/recorder CLIs."""
    if name == "random":
        return RandomPolicy(env.action_space, seed)
    if name == "oracle":
        from baselines.oracle import OraclePolicy

        return OraclePolicy(env)
    if name == "greedy":
        from baselines.greedy import GreedyPolicy

        return GreedyPolicy(env.unwrapped.env_cfg.action_mode)
    raise ValueError(f"unknown baseline: {name!r}")
