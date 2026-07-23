"""Scripted coverage planner for BrickLayerRobot-v0.

The executable spec of good robot play (and the env's solvability tripwire):
place everything reachable bottom-up, then move toward the nearest unfilled
target, repeat. Proves a wall that requires moving is completable.
"""

from __future__ import annotations

import numpy as np

from atrium_sim.blueprint import BrickKind
from atrium_sim.envs.robot_env import Mode


class RobotOraclePolicy:
    def __init__(self, env):
        self.env = env.unwrapped

    def act(self, obs: np.ndarray):
        u = self.env
        target = u._next_place_target()
        if target is not None:  # something reachable -> place it (offset 0, correct kind)
            kind_sign = -0.5 if target.kind == BrickKind.FULL else 0.5
            return (int(Mode.PLACE), np.array([0.0, kind_sign], dtype=np.float32))
        nearest = u._nearest_open()
        if nearest is None:  # nothing left (shouldn't happen before success): stay
            return (int(Mode.PLACE), np.array([0.0, -0.5], dtype=np.float32))
        # move toward the nearest unfilled target
        mode = Mode.MOVE_RIGHT if nearest.x > u.base_x else Mode.MOVE_LEFT
        return (int(mode), np.array([0.0, -0.5], dtype=np.float32))
