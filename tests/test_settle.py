"""Physics validation: stacks rest true, overhangs topple, settling stays cheap."""

import math


from atrium_sim.blueprint import WallSpec, generate_blueprint
from atrium_sim.constants import MAX_SETTLE_SUBSTEPS
from atrium_sim.physics import PhysicsWorld


def build_oracle_wall(spec: WallSpec):
    bp = generate_blueprint(spec)
    world = PhysicsWorld(bp.length)
    substeps = []
    for t in bp.targets:
        bid = world.spawn_brick(t.x, t.kind, t.course)
        assert bid is not None
        n, removed = world.settle(MAX_SETTLE_SUBSTEPS)
        assert not removed
        substeps.append(n)
    return bp, world, substeps


def test_oracle_wall_rests_within_tolerance():
    """Perfect placements must physically rest inside the ±3mm audit plateau.

    Guards the Poly-radius fix (verts = envelope - 2*radius): with wrong verts
    the wall gains ~1mm per course and upper courses drift out of tolerance.
    Slop sag (~0.17mm/course, systematic) is why dy grows with height; at
    collision_slop=0.1 course 5 stays ~1mm - well inside the plateau.
    """
    bp, world, _ = build_oracle_wall(WallSpec(4, 6))
    poses = {p.brick_id: p for p in world.poses()}
    for t in bp.targets:
        p = poses[t.tid]
        d = math.hypot(p.x - t.x, p.y - t.y)
        assert d < 2.0, f"course {t.course}: rest pose {d:.2f}mm from target"
        assert abs(math.degrees(p.theta)) < 0.5
    top = [poses[t.tid] for t in bp.targets if t.course == 5]
    assert all(abs(p.y - 330.0) < 1.5 for p in top)


def test_overhung_brick_topples():
    """A brick with 40mm of unsupported overhang past the wall end must fall."""
    bp = generate_blueprint(WallSpec(4, 1))
    world = PhysicsWorld(bp.length)
    for t in bp.targets:
        world.spawn_brick(t.x, t.kind, t.course)
        world.settle(MAX_SETTLE_SUBSTEPS)
    # course 1: centre 145mm past the last brick's right edge -> ~75% unsupported
    bid = world.spawn_brick(bp.length + 40.0, bp.targets[0].kind, 1)
    world.settle(MAX_SETTLE_SUBSTEPS)
    p = [q for q in world.poses() if q.brick_id == bid][0]
    fell = p.y < 60.0 or abs(math.degrees(p.theta)) > 15.0
    assert fell, f"overhung brick should topple, rests at y={p.y:.1f} theta={math.degrees(p.theta):.1f}"


def test_full_wall_settles_cheaply():
    """Throughput guard on the WORST-case wall (10x6, 63 bodies, one contact
    group): sleep-based settling must keep working as the wall grows - if one
    jittery body kept resetting the group's sleep timer, placements would burn
    the full 600-substep cap and training throughput would collapse."""
    _, _, substeps = build_oracle_wall(WallSpec(10, 6))
    assert substeps[-1] < 150, f"last placement took {substeps[-1]} substeps"
    assert max(substeps) < 300
    assert sum(substeps) / len(substeps) < 100


def test_spawn_probe_never_overlaps():
    """Repeated same-spot placements must stack via the probe, never inject overlap."""
    bp = generate_blueprint(WallSpec(4, 2))
    world = PhysicsWorld(bp.length)
    for _ in range(8):
        world.spawn_brick(bp.length / 2, bp.targets[0].kind, 0)
        world.settle(MAX_SETTLE_SUBSTEPS)
    # the assert inside spawn_brick is the real check; sanity: nothing exploded
    assert world.n_bricks <= 8
    assert all(p.y < 500 for p in world.poses())
