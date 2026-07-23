"""Greedy heuristic: observation-only, no privileged access.

Always places a FULL brick at the env's suggested slot. It fills even courses
(all full) well but botches the half-brick ends of odd courses - so it beats
random comfortably yet leaves clear room for a policy that learns the full/half
decision. A strong-but-beatable anchor for the published curves.
"""

from __future__ import annotations

import numpy as np

from atrium_sim.observations import N_GLOBALS


class GreedyPolicy:
    def __init__(self, action_mode: str = "slot_relative"):
        self.action_mode = action_mode

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.action_mode == "slot_relative":
            return np.array([0.0, -1.0], dtype=np.float32)  # place at slot, always full
        # absolute: aim at the next-slot hint carried in the observation globals
        g = obs[-N_GLOBALS:]
        wall_len, next_slot = g[6], g[8]
        if wall_len <= 0:
            return np.array([0.0, -1.0], dtype=np.float32)
        a0 = 2.0 * next_slot / wall_len - 1.0
        return np.array([np.clip(a0, -1.0, 1.0), -1.0], dtype=np.float32)
