"""Single source of truth for every geometric and physics constant.

Units: millimetres, seconds, kilograms, radians (degrees only in docs/HUD).

Brick format is the Dutch *waalformaat* (the format Monumental's robots lay):
210mm long x 50mm high face, 10mm mortar joints, giving a 220mm horizontal
module and a 60mm course height.
"""

import math

# --- Brick geometry (rendered face, mm) ---
BRICK_FULL_MM = (210.0, 50.0)
BRICK_HALF_MM = (100.0, 50.0)  # half module (110) minus one head joint (10)
JOINT_MM = 10.0
MODULE_MM = 220.0  # brick + head joint
COURSE_MM = 60.0   # brick + bed joint

# --- Mortar-inclusive collision envelopes ---
# There is no wet-mortar simulation: each brick's collision shape is the brick
# inflated by half a joint on every side, so courses stack at exactly 60mm and
# modules abut at 220mm (minus 1mm horizontal clearance to avoid persistent
# flush side-contacts fighting in the solver).
# PyMunk's Poly radius is an *outward* Minkowski offset:
#   effective envelope = verts + 2 * radius.
SHAPE_RADIUS = 0.5
FULL_ENVELOPE = (219.0, 60.0)   # effective collision box of a full brick
HALF_ENVELOPE = (109.0, 60.0)
FULL_VERTS = (FULL_ENVELOPE[0] - 2 * SHAPE_RADIUS, FULL_ENVELOPE[1] - 2 * SHAPE_RADIUS)
HALF_VERTS = (HALF_ENVELOPE[0] - 2 * SHAPE_RADIUS, HALF_ENVELOPE[1] - 2 * SHAPE_RADIUS)

# --- BIM masonry tolerances ---
TOL_MM = 3.0    # position tolerance: full reward plateau half-width
TOL_DEG = 0.5   # levelness tolerance

# --- Physics ---
BRICK_MASS_KG = 2.0             # 210x100x50mm at ~1900 kg/m3
GRAVITY = (0.0, -9810.0)        # mm/s^2
DT = 1.0 / 120.0
MAX_SETTLE_SUBSTEPS = 600       # 5 s sim-time cap per placement
FINAL_SETTLE_SUBSTEPS = 600     # extra 5 s before ANY terminal audit
FRICTION_BRICK = 0.9
FRICTION_GROUND = 1.0
SPACE_DAMPING = 0.90
SPACE_ITERATIONS = 30           # default 10 is too soft for 5-high stacks
SLEEP_TIME_THRESHOLD = 0.15     # s of quiet before a body sleeps (= settle criterion)
IDLE_SPEED_THRESHOLD = 20.0     # mm/s; max hidden creep over the window ~ 3mm ~ tol
COLLISION_SLOP = 0.1            # mm of allowed overlap between objects.

# --- Placement mechanics ---
SPAWN_DROP_MM = 5.0             # brick spawns this far above rest height (the gentle drop)
SPAWN_PROBE_STEP_MM = 10.0      # overlap probe raises spawn y in these steps
DROP_ARM_MARGIN_MM = 60.0       # drop-control mode: the arm "homes" this far above the wall
                                # top; the model picks how far to lower it before releasing,
                                # so impact velocity is an emergent consequence of the fall
OVERHANG_MM = 30.0              # max allowed x-overhang past wall ends when decoding actions
OOB_Y_MM = -100.0               # brick centre below this -> removed, counted as waste
OOB_X_MARGIN_MM = 400.0         # brick centre past wall ends by this -> removed

# --- Matching gate (brick <-> blueprint target) ---
MATCH_GATE_MM = 55.0            # must stay < 60 = half the min same-kind target distance (120)
MATCH_GATE_RAD = math.radians(15.0)

# --- Action decoding ---
# In "slot_relative" mode the agent nudges +-OFFSET_RANGE_MM around the env's
# next open slot. Kept small (15mm) so the whole reachable action range is
# precision-scaled: hitting the +-3mm tolerance needs a[0] within +-0.2 (not
# +-0.006 as on the full wall), and the Gaussian exploration std lands close
# enough to the slot to get graded reward - the gradient that actually drives
# precision. In "absolute" mode a[0] spans the whole wall (the harder,
# ~unlearnable variant kept for comparison).
OFFSET_RANGE_MM = 15.0

# Voussoir tilt nudge (arch placement mode): the agent's dtheta action spans +-this many
# degrees around a voussoir target's intended radial orientation. Grounded in-session (real
# tapered-wedge arch physics spike): a closed, symmetrically-built ring is fully stable through
# +-4deg of off-radial error per voussoir and degrades gracefully (not catastrophically) out to
# +-8deg (still standing at 5/5 seeds) - so the action range covers the whole graceful-
# degradation zone rather than only the perfect point or only the failure zone.
VOUSSOIR_TILT_RANGE_DEG = 8.0

# --- Mobile robot (BrickLayerRobot-v0) ---
# The robot sits on a rail and can only reach slots within REACH_MM of its base.
# REACH < a typical wall, so completing the wall REQUIRES moving - that's the
# hard planning layer (move costs now, pays off via future placements).
REACH_MM = 500.0            # horizontal reach each side of the base
MOVE_STEP_MM = 220.0        # one module per move
MOVE_COST_FRAC = 0.05       # per-move cost as a fraction of one perfect brick's reward

# --- Observation / eval maxima (10 modules x 6 courses) ---
# These bound the base BrickLayer env's fixed C_MAX x S_MAX slot grid. The mobile robot
# no longer uses the grid (it observes a compact SENSOR vector, see robot_env), so these
# caps do not constrain the robot's wall sizes.
L_MAX = 2200.0
H_MAX = 360.0
S_MAX = 11   # slots per course, max (10-module odd course: 2 halves + 9 fulls)
C_MAX = 6

# --- Episode ---
SPARE_FRAC = 0.15               # brick budget = N + max(2, ceil(SPARE_FRAC * N))
MAX_EPISODE_STEPS = 78          # worst case 10x6: N=63, budget=73, +5 belt-and-braces


def wall_length(n_modules: int) -> float:
    """Wall length in mm for n modules (no trailing head joint)."""
    return MODULE_MM * n_modules - JOINT_MM


def brick_budget(n_targets: int) -> int:
    """Placement budget: every target plus ~15% spares (min 2)."""
    return n_targets + max(2, math.ceil(SPARE_FRAC * n_targets))
