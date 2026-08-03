"""Scripted coverage planner for BrickLayerRobot-v0.

The executable spec of good robot play (and the env's solvability tripwire):
place everything reachable bottom-up, then move toward the nearest unfilled
target, repeat. Proves a wall that requires moving is completable.

Real structural arches (BrickKind.VOUSSOIR targets) are offered by the env one at a time,
already in build order (springings-to-keystone) - the oracle just places each at zero
offset/tilt, same as it places any other target at zero offset.
"""

from __future__ import annotations

import numpy as np

from atrium_sim.envs.robot_env import Mode


class RobotOraclePolicy:
    def __init__(self, env):
        self.env = env.unwrapped

    def act(self, obs: np.ndarray):
        u = self.env
        target = u._next_place_target()
        if target is not None:  # something reachable -> place it at zero offset/tilt
            # box[1] (tilt) only matters for VOUSSOIR targets; 0.0 is correct for every kind.
            return (int(Mode.PLACE), np.array([0.0, 0.0, -0.5], dtype=np.float32))
        nearest = u._nearest_open()
        if nearest is None:  # nothing left (shouldn't happen before success): stay
            return (int(Mode.PLACE), np.array([0.0, 0.0, -0.5], dtype=np.float32))
        # move toward the nearest unfilled target
        mode = Mode.MOVE_RIGHT if nearest.x > u.base_x else Mode.MOVE_LEFT
        return (int(mode), np.array([0.0, 0.0, -0.5], dtype=np.float32))
