"""Oracle: places every brick exactly at its blueprint target.

Privileged (reads the env's blueprint and audit directly): it is the
executable spec of perfect play, the reward ceiling, and the standing
integration tripwire - if the oracle can't score ~1.0 within tolerance,
the env or the reward is broken, not the agent.
"""

from __future__ import annotations

import numpy as np

from atrium_sim.blueprint import BrickKind


class OraclePolicy:
    def __init__(self, env):
        self.env = env.unwrapped

    def act(self, obs: np.ndarray) -> np.ndarray:
        u = self.env
        t = u._next_open_target()
        if t is None:  # nothing open in cursor course (shouldn't happen): centre, full
            return np.array([0.0, -1.0], dtype=np.float32)
        a1 = -0.5 if t.kind == BrickKind.FULL else 0.5
        if u.env_cfg.action_mode == "slot_relative":
            a0 = 0.0  # place exactly at the env-suggested slot
        else:
            a0 = 2.0 * t.x / u.blueprint.length - 1.0
        return np.array([a0, a1], dtype=np.float32)
