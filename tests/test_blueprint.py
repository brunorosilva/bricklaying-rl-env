"""Blueprint geometry invariants - the reward's matching logic depends on these."""

import itertools
import math

import numpy as np
import pytest

from atrium_sim.blueprint import (
    EXTRAP_SPECS,
    INTERP_SPECS,
    TRAIN_SPECS,
    BrickKind,
    WallSpec,
    brick_face,
    generate_blueprint,
    sample_spec,
)
from atrium_sim.constants import C_MAX, COURSE_MM, MATCH_GATE_MM, S_MAX, wall_length

ALL_SPECS = TRAIN_SPECS + INTERP_SPECS + EXTRAP_SPECS


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: f"{s.n_modules}x{s.n_courses}")
def test_geometry_invariants(spec: WallSpec):
    bp = generate_blueprint(spec)
    L = wall_length(spec.n_modules)
    assert bp.length == L

    for c in range(spec.n_courses):
        course = bp.course_targets(c)
        assert len(course) <= S_MAX
        # flush wall ends: first brick's left edge at 0, last brick's right edge at L
        first, last = course[0], course[-1]
        assert first.x - brick_face(first.kind)[0] / 2 == pytest.approx(0.0)
        assert last.x + brick_face(last.kind)[0] / 2 == pytest.approx(L)
        # 10mm head joints between neighbours
        for a, b in zip(course, course[1:]):
            gap = (b.x - brick_face(b.kind)[0] / 2) - (a.x + brick_face(a.kind)[0] / 2)
            assert gap == pytest.approx(10.0)
        # course height
        assert all(t.y == pytest.approx(30.0 + COURSE_MM * c) for t in course)
        # halves only at the ends of odd courses
        kinds = [t.kind for t in course]
        if c % 2 == 0:
            assert all(k == BrickKind.FULL for k in kinds)
        else:
            assert kinds[0] == kinds[-1] == BrickKind.HALF
            assert all(k == BrickKind.FULL for k in kinds[1:-1])

    assert spec.n_courses <= C_MAX
    assert bp.n_halves == 2 * (spec.n_courses // 2)
    # targets ordered by (course, slot) and tids consecutive
    assert [t.tid for t in bp.targets] == list(range(bp.n_targets))


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: f"{s.n_modules}x{s.n_courses}")
def test_matching_disjointness_invariant(spec: WallSpec):
    """Min same-KIND target distance must exceed 2x the match gate (120 > 110).

    This is what makes greedy per-target matching equal to optimal assignment:
    every brick can be in-gate for at most one same-kind target. Note the
    invariant is kind-conditional - cross-kind targets get as close as ~81mm.
    """
    bp = generate_blueprint(spec)
    min_same_kind = min(
        math.hypot(a.x - b.x, a.y - b.y)
        for a, b in itertools.combinations(bp.targets, 2)
        if a.kind == b.kind
    )
    assert min_same_kind >= 120.0 > 2 * MATCH_GATE_MM


def test_suites_disjoint_and_sampling():
    assert not set(TRAIN_SPECS) & set(INTERP_SPECS)
    assert not set(TRAIN_SPECS) & set(EXTRAP_SPECS)
    rng = np.random.default_rng(0)
    assert all(sample_spec(rng, "train") in TRAIN_SPECS for _ in range(20))
    assert sample_spec(rng, "interp") in INTERP_SPECS
