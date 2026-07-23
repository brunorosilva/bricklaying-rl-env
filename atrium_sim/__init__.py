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
    # placements + moves; worst case 10x6 ~ budget 200. The env self-terminates
    # on its own step budget first; this TimeLimit is belt-and-braces.
    max_episode_steps=320,
)

__version__ = "0.1.0"
