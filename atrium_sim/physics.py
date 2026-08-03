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
    verts: tuple[tuple[float, float], ...] | None = None
    # The shape's actual LOCAL (body-frame, pre-rotation) vertices, straight from the pymunk
    # shape - always populated. For FULL/HALF this is just the box verts (equivalent to
    # brick_face(kind) but exact); for VOUSSOIR it's the true tapered wedge, which has no
    # fixed (w, h) - the renderer needs this to draw anything other than a rectangle.


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
        self._statics: dict[int, tuple] = {}  # static hard bodies (lintels/cement/roof); never audited
        self._next_id = 0

    # --- spawning -----------------------------------------------------------

    def spawn_brick(self, x: float, kind: BrickKind, course: int,
                    release_y: float | None = None, theta: float = 0.0,
                    rest_y: float | None = None,
                    wedge_verts: tuple[tuple[float, float], ...] | None = None,
                    mass_kg: float | None = None) -> int | None:
        """Place a brick as a dynamic body and let it fall/settle.

        By default it spawns `SPAWN_DROP_MM` above rest height for `course` (the
        gentle drop). If `release_y` is given (drop-control mode), the brick is
        released from that height instead (floored at the gentle height), so its
        impact velocity is an emergent consequence of the fall distance.

        An overlap probe raises the spawn until clear so an overlapping body is
        NEVER injected (deep-overlap resolution flings bricks across the canvas).
        Returns the brick id, or None if no clear height exists below the
        ceiling - the caller charges that as a wasted placement.

        `wedge_verts` (local, centroid-relative, at theta=0 - see atrium_sim.arch) overrides
        the ordinary axis-aligned box shape with an arbitrary convex polygon, for real tapered
        arch voussoirs. `theta` still applies on top (the target's intended orientation plus
        any small agent error), exactly as it already does for the box path. `mass_kg`
        overrides BRICK_MASS_KG (a wedge's true mass scales with its polygon area, not the
        rectangular envelope's). None (both, default) is byte-identical to today.
        """
        mass = BRICK_MASS_KG if mass_kg is None else mass_kg
        if wedge_verts is not None:
            moment = pymunk.moment_for_poly(mass, wedge_verts)
            body = pymunk.Body(mass, moment)
            body.angle = float(theta)
            shape = pymunk.Poly(body, wedge_verts, radius=SHAPE_RADIUS)
        else:
            w, h = _envelope(kind)
            verts = FULL_VERTS if kind == BrickKind.FULL else HALF_VERTS
            moment = pymunk.moment_for_box(mass, (w, h))
            body = pymunk.Body(mass, moment)
            body.angle = float(theta)
            shape = pymunk.Poly.create_box(body, verts, radius=SHAPE_RADIUS)
        shape.friction = FRICTION_BRICK
        shape.elasticity = 0.0

        # rest_y (a voussoir's exact centre y) overrides the course-derived height; for a flat
        # brick rest_y == its target y == COURSE_MM*course + h/2, so this is unchanged. A
        # wedge has no fixed envelope height, so rest_y is mandatory whenever wedge_verts is
        # given (the caller - atrium_sim.arch's targets always set it).
        if rest_y is None and wedge_verts is not None:
            raise ValueError("wedge_verts requires an explicit rest_y")
        gentle_y = (rest_y if rest_y is not None else COURSE_MM * course + h / 2.0) + SPAWN_DROP_MM
        # release_y=None -> identical to the original fixed gentle drop; otherwise
        # honor the requested height but never start below gentle (the probe only
        # ever raises y, preserving the never-inject-overlap guarantee).
        y = gentle_y if release_y is None else max(release_y, gentle_y)
        # constant probe headroom (2 courses) ABOVE the requested spawn height, so the
        # overlap probe can raise the brick a little to find a clear spot. Size-agnostic:
        # was a fixed H_MAX+120 = 480mm that gave shrinking headroom as courses rose and
        # hit zero at ~course 8, physically capping every wall at 8 courses.
        ceiling = y + 120.0
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

    def spawn_static_body(self, verts_mm, kind: str = "cement", sensor: bool = False) -> int:
        """A monolithic STATIC obstacle (lintel / cement curve / roof cap): bricks rest on and
        abut it, it never moves or topples, and it is NOT a brick - it never enters poses() or
        the audit. `verts_mm` is a convex polygon in world mm. If `sensor`, the shape is
        non-colliding (a decorative voussoir that bears on the piers without flinging them).
        Returns its id."""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        shape = pymunk.Poly(body, [(float(x), float(y)) for x, y in verts_mm], radius=SHAPE_RADIUS)
        shape.friction = FRICTION_BRICK
        shape.elasticity = 0.0
        shape.sensor = bool(sensor)
        self.space.add(body, shape)
        sid = self._next_id
        self._next_id += 1
        self._statics[sid] = (body, shape, kind)
        return sid

    def remove_static_body(self, sid: int) -> None:
        """Strike a static body (a temporary arch centering) - the missing counterpart to
        `spawn_static_body`. Once removed, whatever it was holding up (an arch ring) either
        stands on its own or doesn't; this is the moment that decides."""
        body, shape, _kind = self._statics.pop(sid)
        self.space.remove(body, shape)

    def contact_normal_impulse(self, brick_id: int) -> float:
        """Sum of |horizontal component| of the normal impulse across every live contact on
        this body, divided by DT -> an instantaneous force estimate (N, since masses are kg
        and mm/s^2 accelerations here are actually consistent with N when mass is in kg and
        distances in mm only if impulse is in kg*mm/s - this is kg*mm/s^2 = mN; callers divide
        by 1000 for N, matching the in-session spike's convention). Used to measure REAL
        springing thrust (and, via the caller pairing this with contact-point positions, the
        thrust-line eccentricity for the middle-third check) instead of a closed-form guess."""
        body = self._bricks[brick_id].body
        total = 0.0

        def _accum(arbiter: pymunk.Arbiter) -> None:
            nonlocal total
            total += abs(arbiter.total_impulse.x) / DT

        body.each_arbiter(_accum)
        return total

    def hard_poses(self) -> list[tuple[int, str, list[tuple[float, float]]]]:
        """(id, kind, world-space vertices) per static body - for the renderer only."""
        return [(sid, kind, [tuple(body.local_to_world(v)) for v in shape.get_vertices()])
                for sid, (body, shape, kind) in self._statics.items()]

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
            BrickPose(bid, b.body.position.x, b.body.position.y, b.body.angle, b.kind,
                      verts=tuple(b.shape.get_vertices()))
            for bid, b in self._bricks.items()
        ]

    def positions(self) -> dict[int, tuple[float, float]]:
        """Snapshot of centres, for disturbance measurement across a settle."""
        return {bid: (b.body.position.x, b.body.position.y) for bid, b in self._bricks.items()}

    @property
    def n_bricks(self) -> int:
        return len(self._bricks)
