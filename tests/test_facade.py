"""FacadePlan IR + deterministic tiler (offline, no network)."""

import pytest

from atrium_sim.facade import FacadePanel, FacadePlan, Opening, tile_facade
from atrium_sim.blueprint import WallSpec


def _brick_cell_count(cols, rows, openings):
    void = set()
    for o in openings:
        for r in range(o.row, o.row + o.n_rows):
            for c in range(o.col, o.col + o.n_cols):
                void.add((c, r))
    return cols * rows - len(void)


def test_tiling_is_an_exact_partition():
    """Panels exactly cover the brick cells (grid minus openings): total panel area ==
    brick-cell count, no cell claimed twice, none inside an opening."""
    cols, rows = 20, 12
    openings = (Opening("window", 3, 2, 4, 5), Opening("door", 12, 0, 3, 8))
    panels = tile_facade(cols, rows, openings)
    area = sum(p.spec.n_modules * p.spec.n_courses for p in panels)
    assert area == _brick_cell_count(cols, rows, openings)
    # reconstruct the covered cells and check for exactness / no double-claim
    covered = set()
    for p in panels:
        for c in range(p.origin_col, p.origin_col + p.spec.n_modules):
            for r in range(p.origin_row, p.origin_row + p.spec.n_courses):
                assert (c, r) not in covered, "double-claimed cell"
                covered.add((c, r))
    for o in openings:
        for c in range(o.col, o.col + o.n_cols):
            for r in range(o.row, o.row + o.n_rows):
                assert (c, r) not in covered, "panel covers an opening"


def test_from_perception_validates_and_builds():
    plan = FacadePlan.from_perception(
        "mock.jpg", 55, 44,
        [Opening("window", 3, 10, 10, 15), Opening("door", 24, 0, 5, 20)],
        notes="left gable half-timbered",
    )
    assert plan.panels and plan.n_bricks == sum(
        bp.n_targets for bp, _ in plan.blueprints()
    )
    assert plan.validate() is plan  # no raise


def test_validate_catches_overlap_and_out_of_grid():
    # two overlapping panels
    bad = FacadePlan("x", 10, 10, (), (
        FacadePanel(WallSpec(5, 5), 0, 0), FacadePanel(WallSpec(5, 5), 3, 3)))
    with pytest.raises(ValueError):
        bad.validate()
    # panel out of grid
    oob = FacadePlan("x", 5, 5, (), (FacadePanel(WallSpec(9, 2), 0, 0),))
    with pytest.raises(ValueError):
        oob.validate()
    # opening out of grid
    with pytest.raises(ValueError):
        FacadePlan("x", 5, 5, (Opening("window", 4, 4, 5, 5),), ()).validate()


def test_openings_are_clamped_to_grid():
    # an opening spilling past the edge is clamped, not rejected
    plan = FacadePlan.from_perception("x", 10, 10, [Opening("window", 8, 8, 5, 5)])
    o = plan.openings[0]
    assert o.col + o.n_cols <= 10 and o.row + o.n_rows <= 10


def test_json_round_trip():
    plan = FacadePlan.from_perception(
        "x", 30, 20, [Opening("window", 5, 5, 6, 6), Opening("door", 20, 0, 4, 10)])
    assert FacadePlan.from_json(plan.to_json()).to_json() == plan.to_json()


@pytest.mark.parametrize("style", ["semicircular", "segmental"])
def test_crown_packing_lands_on_the_course_line(style):
    """crown_packing_hard_body's top face must land EXACTLY on the crown course's bottom
    edge (COURSE_MM * crown_course), with no vertical inset. A vertical inset here would
    leave every crown-course brick's target resting on nothing, which the spawn probe then
    "fixes" by lifting the brick into a tilt bad enough to blow the match-angle gate
    (confirmed: a -6mm inset was capping the whole crown course out of tolerance)."""
    from atrium_sim.constants import COURSE_MM

    o = Opening("window", col=2, row=0, n_cols=3, n_rows=8, has_lintel=False, has_sill=False,
                arch_style=style, arch_ring_courses=2)
    plan = FacadePlan.from_perception("t", 7, 9, [o])
    regions = list(plan.arch_regions())
    assert len(regions) == 1
    region = regions[0]
    pkg = region.crown_packing_hard_body()
    assert pkg is not None, f"{style}: expected the ring to fall short of the crown line"
    top_y = max(y for _, y in pkg.verts_mm)
    assert top_y == pytest.approx(COURSE_MM * region.crown_course)
