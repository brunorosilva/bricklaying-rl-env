"""Real structural arch physics: pins the in-session validation spike as permanent tests.

These build voussoir rings DIRECTLY via atrium_sim.arch + atrium_sim.physics (not through
the full robot_env/facade stack) - the same methodology used to de-risk the whole feature
before writing a line of env-integration code: tapered wedges (not rectangular bricks on a
curve), a real centering, a real strike. What's pinned here:

- a correctly closed, symmetrically-built ring survives being struck (a few mm of settlement,
  not a collapse);
- an asymmetric (left-to-right) build order collapses the SAME final geometry on striking -
  construction SEQUENCE matters, not just the finished shape;
- a ring missing any one voussoir (keystone or a springer) collapses on striking;
- this holds across all three arch kinds (semicircular, segmental, jack);
- min_thickness_ok() lands on the Couplet/Heyman/Milankovitch limit (t/R ~ 0.1075).
"""

from __future__ import annotations

import pytest

from atrium_sim.arch import (
    ArchSpec,
    abutment_wedge_verts,
    arch_wedges,
    build_order,
    centering_polygon,
    ring_drift,
    survived,
    wedge_verts_mass_kg,
)
from atrium_sim.blueprint import BrickKind
from atrium_sim.constants import COURSE_MM
from atrium_sim.physics import PhysicsWorld

SPAN = 660.0
PIER_COURSES = 4
ORIGIN_X = 1000.0


def _build_piers(world: PhysicsWorld, span: float, pier_modules: int = 3) -> None:
    for side in (-1, 1):
        x_edge = ORIGIN_X + side * span / 2.0
        for c in range(PIER_COURSES):
            for k in range(pier_modules):
                bx = x_edge + side * (219.0 * (k + 0.5))
                assert world.spawn_brick(bx, BrickKind.FULL, c) is not None
                world.settle(200)


def _build_and_strike(spec: ArchSpec, *, symmetric_order: bool = True, strike: bool = True,
                       drop_index: int | None = None):
    """Build piers + skewback + centering, place every voussoir (in the requested order,
    optionally dropping one), strike, settle, and report (drift_mm, tilt_deg, ok, n_failed)."""
    world = PhysicsWorld(3000.0)
    _build_piers(world, spec.span_mm)
    spring_y = COURSE_MM * PIER_COURSES
    for side in (-1, 1):
        v = abutment_wedge_verts(spec, side)
        if v:
            world.spawn_static_body([(ORIGIN_X + x, spring_y + y) for x, y in v], "skewback")
    wedges = arch_wedges(spec)
    cent_world = [(ORIGIN_X + x, spring_y + y) for x, y in centering_polygon(spec)]
    cent_id = world.spawn_static_body(cent_world, "centering", sensor=False)

    order = list(build_order(spec.n_voussoirs)) if symmetric_order else list(range(spec.n_voussoirs))
    if drop_index is not None:
        order = [k for k in order if k != drop_index]

    placed: dict[int, int] = {}
    n_failed = 0
    for k in order:
        w = wedges[k]
        bid = world.spawn_brick(
            ORIGIN_X + w.x, BrickKind.VOUSSOIR, 0, theta=w.theta, rest_y=spring_y + w.y,
            wedge_verts=w.verts, mass_kg=wedge_verts_mass_kg(w.verts),
        )
        if bid is None:
            n_failed += 1
            continue
        placed[k] = bid
        world.settle(300)

    def _snapshot():
        poses = {p.brick_id: p for p in world.poses()}
        return {
            k: (poses[bid].x, poses[bid].y, poses[bid].theta) if bid in poses else (1e6, 1e6, 0.0)
            for k, bid in placed.items()
        }

    before = _snapshot()
    if strike:
        world.remove_static_body(cent_id)
    world.settle(1800)
    after = _snapshot()

    drift, tilt = ring_drift(before, after)
    ok = survived(drift, tilt) and n_failed == 0
    return drift, tilt, ok, n_failed


SEMICIRCULAR = ArchSpec(kind="semicircular", span_mm=SPAN, rise_mm=SPAN / 2.0,
                         ring_depth_mm=210.0, n_voussoirs=11)
SEGMENTAL = ArchSpec(kind="segmental", span_mm=SPAN, rise_mm=110.0,
                      ring_depth_mm=210.0, n_voussoirs=9)
JACK = ArchSpec(kind="jack", span_mm=SPAN, rise_mm=1.0, ring_depth_mm=210.0, n_voussoirs=9)


@pytest.mark.parametrize("spec", [SEMICIRCULAR, SEGMENTAL, JACK],
                         ids=["semicircular", "segmental", "jack"])
def test_symmetric_build_survives_the_strike(spec):
    """The one build order validated to survive: centering in place, voussoirs placed
    springings-to-keystone, then struck. Small settlement, no collapse."""
    drift, tilt, ok, failed = _build_and_strike(spec)
    assert failed == 0
    assert ok, f"expected survival, got drift={drift:.1f}mm tilt={tilt:.1f}deg"
    assert drift < 20.0
    assert tilt < 10.0


@pytest.mark.parametrize("spec", [SEMICIRCULAR, SEGMENTAL, JACK],
                         ids=["semicircular", "segmental", "jack"])
def test_asymmetric_build_order_collapses(spec):
    """The IDENTICAL final ring geometry, built left-to-right instead of springings-to-
    keystone, does not survive striking (some voussoirs even fail to settle mid-build) -
    construction sequence is a real physical constraint, not just final geometry."""
    drift, tilt, ok, failed = _build_and_strike(spec, symmetric_order=False)
    assert not ok


@pytest.mark.parametrize("spec", [SEMICIRCULAR, SEGMENTAL],
                         ids=["semicircular", "segmental"])
def test_missing_keystone_collapses(spec):
    keystone = (spec.n_voussoirs - 1) // 2
    drift, tilt, ok, failed = _build_and_strike(spec, drop_index=keystone)
    assert not ok


@pytest.mark.parametrize("spec", [SEMICIRCULAR, SEGMENTAL],
                         ids=["semicircular", "segmental"])
def test_missing_springer_collapses(spec):
    drift, tilt, ok, failed = _build_and_strike(spec, drop_index=0)
    assert not ok


def test_not_struck_trivially_stands():
    """With the centering still in place (never struck), the ring is supported regardless of
    build order - a sanity check that _build_and_strike's harness itself is sound."""
    drift, tilt, ok, failed = _build_and_strike(SEMICIRCULAR, strike=False)
    assert ok
    assert drift < 1.0


def test_min_thickness_matches_couplet_heyman_limit():
    """t/R ~ 0.1075 is the Couplet/Heyman/Milankovitch minimum for a semicircular arch to
    admit any valid thrust line at all. Comfortably above (0.182) must pass; comfortably
    below (0.106, i.e. below the true limit) must fail the safety-margined check."""
    r = SPAN / 2.0
    ok_spec = ArchSpec(kind="semicircular", span_mm=SPAN, rise_mm=r,
                       ring_depth_mm=0.182 * r, n_voussoirs=11)
    bad_spec = ArchSpec(kind="semicircular", span_mm=SPAN, rise_mm=r,
                        ring_depth_mm=0.106 * r, n_voussoirs=11)
    assert ok_spec.min_thickness_ok()
    assert not bad_spec.min_thickness_ok()


def test_thin_ring_below_min_thickness_collapses():
    """The min_thickness_ok() check isn't just bookkeeping - a ring built at t/R below the
    Heyman limit genuinely collapses in the physics, even with a perfect symmetric build and
    a perfectly closed ring."""
    r = SPAN / 2.0
    thin = ArchSpec(kind="semicircular", span_mm=SPAN, rise_mm=r,
                    ring_depth_mm=0.106 * r, n_voussoirs=11)
    assert not thin.min_thickness_ok()
    drift, tilt, ok, failed = _build_and_strike(thin)
    assert not ok


def test_odd_voussoir_count_enforced():
    with pytest.raises(ValueError):
        ArchSpec(kind="semicircular", span_mm=SPAN, rise_mm=SPAN / 2.0,
                 ring_depth_mm=210.0, n_voussoirs=10)


def test_build_order_is_symmetric_springings_to_keystone():
    order = build_order(11)
    assert len(order) == 11 and set(order) == set(range(11))
    assert order[-1] == 5  # the keystone (dead centre) is placed LAST
    # each successive pair alternates left/right springings inward
    assert order[:2] in ((0, 10), (10, 0))
