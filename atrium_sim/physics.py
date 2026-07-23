"""PyMunk world: rigid-body bricks, gravity, settling.

Design notes (the *why* behind the numbers, see constants.py for values):

- Mortar is not simulated as a material. Each brick's collision shape is the
  brick inflated by half a joint on every side ("mortar-inclusive envelope"),
  so courses stack at exactly 60mm and modules abut at 220mm. High friction
  stands in for fresh-mortar tack, but there is NO adhesion: an overhung or
  crooked brick will slide and topple - exactly the failure mode we want
  physics to produce.
- Sleeping IS the settle criterion. A placement is settled when every brick
  in the space is asleep (0.15s below the idle speed threshold), capped at
  MAX_SETTLE_SUBSTEPS. This doubles as the performance win: a sleeping wall
  costs almost nothing to step.
- Determinism: Chipmunk has no internal RNG. Fixed dt, deterministic body
  insertion order and a fresh Space per episode give bit-identical
  trajectories on the same platform/pymunk build.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import pymunk

from atrium_sim.blueprint import BrickKind
from atrium_sim.constants import (
    BRICK_MASS_KG,
    COLLISION_SLOP,
    COURSE_MM,
    DT,
    FRICTION_BRICK,
    FRICTION_GROUND,
    FULL_ENVELOPE,
    FULL_VERTS,
    GRAVITY,
    HALF_ENVELOPE,
    HALF_VERTS,
    H_MAX,
    IDLE_SPEED_THRESHOLD,
    OOB_X_MARGIN_MM,
    OOB_Y_MM,
    SHAPE_RADIUS,
    SLEEP_TIME_THRESHOLD,
    SPACE_DAMPING,
    SPACE_ITERATIONS,
    SPAWN_DROP_MM,
    SPAWN_PROBE_STEP_MM,
)


class BrickPose(NamedTuple):
    brick_id: int
    x: float
    y: float
    theta: float  # radians, unbounded (reward folds it)
    kind: BrickKind


class _Brick(NamedTuple):
    body: pymunk.Body
    shape: pymunk.Poly
    kind: BrickKind


def _envelope(kind: BrickKind) -> tuple[float, float]:
    return FULL_ENVELOPE if kind == BrickKind.FULL else HALF_ENVELOPE


class PhysicsWorld:
    """One episode's physics. Build fresh every reset - never reuse a Space."""

    def __init__(self, wall_length: float):
        self.wall_length = wall_length
        space = pymunk.Space()
        space.gravity = GRAVITY
        space.damping = SPACE_DAMPING
        space.iterations = SPACE_ITERATIONS
        space.sleep_time_threshold = SLEEP_TIME_THRESHOLD
        space.idle_speed_threshold = IDLE_SPEED_THRESHOLD
        space.collision_slop = COLLISION_SLOP
        ground = pymunk.Poly(
            space.static_body,
            [(-300.0, -50.0), (wall_length + 300.0, -50.0),
             (wall_length + 300.0, 0.0), (-300.0, 0.0)],
        )
        ground.friction = FRICTION_GROUND
        ground.elasticity = 0.0
        space.add(ground)
        self.space = space
        self._bricks: dict[int, _Brick] = {}  # insertion-ordered -> deterministic solver order
        self._next_id = 0

    # --- spawning -----------------------------------------------------------

    def spawn_brick(self, x: float, kind: BrickKind, course: int,
                    release_y: float | None = None) -> int | None:
        """Place a brick as a dynamic body and let it fall/settle.

        By default it spawns `SPAWN_DROP_MM` above rest height for `course` (the
        gentle drop). If `release_y` is given (drop-control mode), the brick is
        released from that height instead (floored at the gentle height), so its
        impact velocity is an emergent consequence of the fall distance.

        An overlap probe raises the spawn until clear so an overlapping body is
        NEVER injected (deep-overlap resolution flings bricks across the canvas).
        Returns the brick id, or None if no clear height exists below the
        ceiling - the caller charges that as a wasted placement.
        """
        w, h = _envelope(kind)
        verts = FULL_VERTS if kind == BrickKind.FULL else HALF_VERTS
        moment = pymunk.moment_for_box(BRICK_MASS_KG, (w, h))
        body = pymunk.Body(BRICK_MASS_KG, moment)
        shape = pymunk.Poly.create_box(body, verts, radius=SHAPE_RADIUS)
        shape.friction = FRICTION_BRICK
        shape.elasticity = 0.0

        gentle_y = COURSE_MM * course + h / 2.0 + SPAWN_DROP_MM
        # release_y=None -> identical to the original fixed gentle drop; otherwise
        # honor the requested height but never start below gentle (the probe only
        # ever raises y, preserving the never-inject-overlap guarantee).
        y = gentle_y if release_y is None else max(release_y, gentle_y)
        ceiling = H_MAX + 120.0
        while y <= ceiling:
            body.position = (x, y)
            if not self.space.shape_query(shape):
                break
            y += SPAWN_PROBE_STEP_MM
        else:
            return None  # no clear height below the ceiling: skip, charge waste

        assert not self.space.shape_query(shape), "spawn probe must never inject overlap"
        self.space.add(body, shape)
        brick_id = self._next_id
        self._next_id += 1
        self._bricks[brick_id] = _Brick(body, shape, kind)
        return brick_id

    # --- settling -----------------------------------------------------------

    def settle(
        self,
        max_substeps: int,
        frame_cb: Callable[[], None] | None = None,
        frame_stride: int = 4,
    ) -> tuple[int, list[int]]:
        """Step physics until every brick sleeps (or the cap hits).

        Out-of-bounds bricks are removed *inside* the loop - a brick falling
        off the canvas never sleeps and would otherwise burn the whole cap.
        Returns (substeps used, ids removed as off-canvas waste).
        """
        removed: list[int] = []
        substeps = 0
        for i in range(max_substeps):
            self.space.step(DT)
            substeps = i + 1
            if frame_cb is not None and i % frame_stride == 0:
                frame_cb()
            if i % 12 == 11:
                removed.extend(self._remove_out_of_bounds())
            if all(b.body.is_sleeping for b in self._bricks.values()):
                break
        removed.extend(self._remove_out_of_bounds())
        return substeps, removed

    def _remove_out_of_bounds(self) -> list[int]:
        gone = [
            bid
            for bid, b in self._bricks.items()
            if b.body.position.y < OOB_Y_MM
            or not (-OOB_X_MARGIN_MM <= b.body.position.x <= self.wall_length + OOB_X_MARGIN_MM)
        ]
        for bid in gone:
            brick = self._bricks.pop(bid)
            self.space.remove(brick.body, brick.shape)
        return gone

    # --- state --------------------------------------------------------------

    def poses(self) -> list[BrickPose]:
        return [
            BrickPose(bid, b.body.position.x, b.body.position.y, b.body.angle, b.kind)
            for bid, b in self._bricks.items()
        ]

    def positions(self) -> dict[int, tuple[float, float]]:
        """Snapshot of centres, for disturbance measurement across a settle."""
        return {bid: (b.body.position.x, b.body.position.y) for bid, b in self._bricks.items()}

    @property
    def n_bricks(self) -> int:
        return len(self._bricks)
