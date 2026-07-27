"""atrium-sim: a physics-based bricklaying RL environment.

Importing this package registers `atrium_sim/BrickLayer-v0`.
"""

from gymnasium.envs.registration import register

from atrium_sim.constants import MAX_EPISODE_STEPS

register(
    id="atrium_sim/BrickLayer-v0",
    entry_point="atrium_sim.envs.bricklayer_env:BrickLayerEnv",
    # Worst case (10 modules x 6 courses): N=63 targets, budget 73, +5 headroom.
    # The env self-terminates on budget first; this TimeLimit is belt-and-braces.
    max_episode_steps=MAX_EPISODE_STEPS,
)

register(
    id="atrium_sim/BrickLayerRobot-v0",
    entry_point="atrium_sim.envs.robot_env:BrickLayerRobotEnv",
    # The env's OWN budget is size-aware (robot_env.py: budget + 2*n_courses*traversal)
    # and self-terminates first - e.g. ~469 steps for a 12x12 wall, ~900 for 20x14. The
    # old fixed 320 cap truncated big walls before the internal budget was spent, so the
    # top courses were never reached. 2000 is a belt-and-braces bound above the largest
    # internal budget; the size-aware budget governs.
    max_episode_steps=2000,
)

__version__ = "0.1.0"
