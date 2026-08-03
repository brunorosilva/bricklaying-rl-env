"""FacadePlan: turn a VLM's perception of a brick facade into buildable panels.

Division of labour (deliberate): the VLM does *perception* — it estimates the module
grid (1 col = 220mm module, 1 row = 60mm course, origin = bottom-left of the brickwork)
and locates the openings (windows/doors) as grid rectangles. The *geometry* — carving
the remaining brickwork into non-overlapping running-bond rectangles — is done
DETERMINISTICALLY here, so we never ask the model to produce a valid tiling (which it
can't reliably do). Approximate-but-cohesive perception + exact tiling.

Import-light (no torch/gym), same tier as blueprint.py, so the audit/render/oracle can
consume facades standalone.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from atrium_sim.arch import (
    ArchSpec,
    abutment_wedge_verts,
    arch_wedges,
    centering_polygon,
)
from atrium_sim.blueprint import (
    BrickKind,
    BrickTarget,
    Blueprint,
    WallSpec,
    brick_face,
    generate_blueprint,
)
from atrium_sim.constants import COURSE_MM, MODULE_MM

ARCH_STYLES = ("flat", "lintel_soldier", "semicircular", "segmental", "jack")
STRUCTURAL_ARCH_STYLES = ("semicircular", "segmental", "jack")

PACKING_CLEARANCE_MM = 6.0
# Crown/spandrel packing edges that face a FUTURE DYNAMIC SPAWN (the adjacent pier brick, or a
# voussoir) are inset by this much rather than placed exactly flush with the boundary they're
# closing. Discovered in-session: a panel's edge-column brick target is centred using the SAME
# per-panel "-5mm" running-bond offset as every other column (`_course_targets`'s 105.0/215.0
# constants), so its own half-width (FULL_ENVELOPE[0]/2 = 109.5mm) overruns the panel's nominal
# edge by 109.5 - (MODULE_MM/2 - 5) = 4.5mm - harmless against another ordinary panel (which
# overruns its own edge by the same amount, so two abutting panels' bricks still leave the
# normal ~1mm running-bond gap), but enough to physically foul a packing block whose edge sits
# EXACTLY at that same panel boundary. Because `physics.spawn_brick`'s overlap probe only
# climbs (never widens), that fouled spawn blocked EVERY height up to the packing's own top,
# stranding the pier brick spawning next to it ~30-60mm above its target (permanently out of
# tolerance, never falling, never matching - the actual root cause of the wall stalling at the
# springing course, not a support gap at all). 6mm clears the 4.5mm worst case with margin; the
# resulting sliver of unfilled space between packing and pier is physically inert (nothing needs
# to bridge across it).


@dataclass(frozen=True)
class Opening:
    """A void in the brick field: cols [col, col+n_cols), rows [row, row+n_rows)."""
    kind: str
    col: int
    row: int
    n_cols: int
    n_rows: int
    has_lintel: bool = True    # a spanning hard beam carries the brickwork ABOVE the opening
                               # (styles "flat"/"lintel_soldier" only - a real arch IS its own head)
    has_sill: bool = False     # a spanning hard ledge at the base (windows)
    arch_style: str = "flat"   # "flat" (plain lintel) | "lintel_soldier" (a flat cement lintel
                               # + a cosmetic fanned near-vertical soldier course, NON-structural -
                               # the control: BIA describes this as standard modern practice
                               # when "the structural resistance of the arch is neglected") |
                               # "semicircular" / "segmental" / "jack" (a REAL structural voussoir
                               # ring on a centering - see atrium_sim.arch)
    arch_rise_ratio: float = 0.5   # rise / span, for the two circular-arc styles (0.5 =
                                   # semicircular); ignored for "jack" and non-arch styles
    arch_ring_courses: int = 4     # ring depth, in whole courses (BIA: a jack arch's depth is
                                   # typically 3-5 courses of the surrounding brickwork)
    arch_n_voussoirs: int | None = None   # None = auto (odd count sized to the span)


@dataclass(frozen=True)
class HardBody:
    """A resolved STATIC (never-audited) element in GLOBAL mm - a lintel/sill/cement/roof the
    bricks rest on and abut. Spawned into the world once the build completes `trigger_course`
    ('the opening's lintel appears at the exact brick level', like real construction)."""
    kind: str
    verts_mm: tuple[tuple[float, float], ...]
    trigger_course: int
    static: bool = True


@dataclass(frozen=True)
class ArchRegion:
    """A real structural arch, resolved to GLOBAL mm - ready for the robot to build voussoir by
    voussoir on a centering. `origin_x`/`spring_y` place the arch's LOCAL frame (x=0 at the
    centerline, y=0 at the spring line - see atrium_sim.arch) in the facade's global
    coordinates."""
    opening_index: int
    spec: ArchSpec
    origin_x: float
    spring_y: float
    springing_course: int   # course at which the centering/skewback wedges may be spawned
    crown_course: int       # course at which flat brickwork resumes above (== the opening's
                            # own top - the tiler already carves the void up to exactly here)
    void_x0: float          # the TILER's own QUANTIZED (whole-module) void bounds, in GLOBAL
    void_x1: float          # mm (see `_arch_void_bounds_mm`) - crown/spandrel packing fill
                            # exactly to these edges, not the ring's own (narrower) raw reach,
                            # so they never leave a gap short of where the tiler actually
                            # stopped excluding flat pier targets.

    def voussoir_targets(self, tid_start: int) -> tuple[BrickTarget, ...]:
        """Every voussoir as a BrickTarget in GLOBAL mm, geometric (left-to-right) order, tid
        assigned from tid_start. `slot` encodes BUILD order (springings-to-keystone - see
        atrium_sim.arch.build_order), not left-to-right position, so the env can offer them in
        the physically-required sequence without re-deriving it."""
        from atrium_sim.arch import build_order

        wedges = arch_wedges(self.spec)
        order = build_order(self.spec.n_voussoirs)
        build_slot = {idx: slot for slot, idx in enumerate(order)}
        return tuple(
            BrickTarget(
                tid=tid_start + w.index, course=self.springing_course, slot=build_slot[w.index],
                x=self.origin_x + w.x, y=self.spring_y + w.y, kind=BrickKind.VOUSSOIR,
                theta=w.theta, wedge_verts=w.verts, arch_id=self.opening_index,
            )
            for w in wedges
        )

    def centering_hard_body(self) -> HardBody:
        verts = tuple(
            (self.origin_x + x, self.spring_y + y) for x, y in centering_polygon(self.spec)
        )
        return HardBody("centering", verts, trigger_course=max(0, self.springing_course - 1))

    def abutment_hard_bodies(self) -> tuple[HardBody, ...]:
        """Permanent (never struck) skewback filler blocks - see
        atrium_sim.arch.abutment_wedge_verts for why they're structurally necessary."""
        out = []
        for side in (-1, 1):
            v = abutment_wedge_verts(self.spec, side)
            if v:
                verts = tuple((self.origin_x + x, self.spring_y + y) for x, y in v)
                out.append(HardBody("skewback", verts, trigger_course=max(0, self.springing_course - 1)))
        return tuple(out)

    def crown_packing_hard_body(self) -> HardBody | None:
        """A permanent (never struck) SPANDREL FILL closing any gap between the ring's actual
        extrados apex and the crown course, where flat brickwork resumes above - the ring's
        rise is quantised to whole courses at the springing, but its true outer height (rise +
        ring depth) doesn't generally land exactly on a course boundary. Without this, the
        first flat brick placed directly above an otherwise-correct, successfully-struck ring
        can find nothing directly beneath it and topple repeatedly. Real masonry fills this
        same space with SPANDREL packing between the haunches and the next course line. None
        if the ring already reaches (or exceeds) the crown course.

        Width matches the tiler's own QUANTIZED void bounds (`void_x0`/`void_x1`), NOT the
        opening's bare span and NOT the ring's raw (unquantized) reach - the tiler carves flat
        pier targets away up to those exact whole-module edges (the ring oversails onto the
        pier near the springing, then that raw bound is rounded OUT to whole columns), so the
        crown-course brick directly above can land anywhere within them, not just within the
        bare span or the ring's own unrounded footprint."""
        wedges = arch_wedges(self.spec)
        ring_top_y = max(
            self.spring_y + w.y + (lx * math.sin(w.theta) + ly * math.cos(w.theta))
            for w in wedges for lx, ly in w.verts
        )
        crown_y = COURSE_MM * self.crown_course - PACKING_CLEARANCE_MM
        if ring_top_y >= crown_y - 1.0:
            return None
        x0, x1 = self.void_x0, self.void_x1
        return HardBody("cement", ((x0, ring_top_y), (x1, ring_top_y), (x1, crown_y), (x0, crown_y)),
                        trigger_course=max(0, self.crown_course - 1))

    def spandrel_hard_bodies(self) -> tuple[HardBody, ...]:
        """Permanent (never struck) fill closing the gap, AT EVERY COURSE from the springing to
        the crown, between the ring's actual per-course reach (`ring_row_spans`, narrower than
        the tiler's void at every row except the single widest one) and the tiling void's own
        QUANTIZED (whole-module) edges (`void_x0`/`void_x1` - the tiler deliberately uses one
        uniform, widest-row-sized void rather than tapering course by course, to avoid two
        previously-tried and reverted failure modes: broken running bond from course-tall
        panels, and unsupported overhangs from a non-monotonic taper). Without this, every
        course except the ring's single widest row leaves the adjacent pier brick with nothing
        directly beneath it at that row - exactly the crown-apex gap `crown_packing_hard_body`
        already fixes, just recurring at every course instead of only the top one. Real masonry
        fills this same space with spandrel packing between the ring's haunches and the
        surrounding coursing - this is that packing, not a workaround.

        A row with more than one disjoint interval (e.g. two springer slivers near the springing,
        with the opening's own genuine clear void between them) only gets filled OUTSIDE the
        outermost interval on each side - the interior gap is the real opening, left alone."""
        from atrium_sim.arch import ring_row_spans

        spans = ring_row_spans(self.spec)
        n_rows = self.crown_course - self.springing_course
        out = []
        for r in range(n_rows):
            intervals = spans.get(r)
            if not intervals:
                continue
            ring_lo = self.origin_x + min(lo for lo, _ in intervals)
            ring_hi = self.origin_x + max(hi for _, hi in intervals)
            y0 = self.spring_y + COURSE_MM * r
            y1 = y0 + COURSE_MM
            trigger = max(0, self.springing_course + r - 1)
            # inset BOTH edges (see PACKING_CLEARANCE_MM): the outer edge would otherwise sit
            # exactly flush with the pier brick spawning just outside it, the inner edge
            # exactly flush with the voussoir that will occupy ring_lo/ring_hi.
            x0, x1 = self.void_x0 + PACKING_CLEARANCE_MM, ring_lo - PACKING_CLEARANCE_MM
            if x1 - x0 > 1.0:
                out.append(HardBody("cement", ((x0, y0), (x1, y0), (x1, y1), (x0, y1)), trigger))
            x0, x1 = ring_hi + PACKING_CLEARANCE_MM, self.void_x1 - PACKING_CLEARANCE_MM
            if x1 - x0 > 1.0:
                out.append(HardBody("cement", ((x0, y0), (x1, y0), (x1, y1), (x0, y1)), trigger))
        return tuple(out)


ARCH_PLAN_SPECS: tuple[tuple[str, int, int, int, int, int, int, int], ...] = (
    # (arch_style, col, row, n_cols, n_rows, arch_ring_courses, grid_cols, grid_rows)
    # Each entry is a hand-validated (oracle: ring_closure=1, strike_survival=1, frac_filled
    # >= 0.95) small facade with ONE structural arch - the curriculum's arch-plan analogue of
    # blueprint.SIZE_LADDER's flat WallSpecs. "jack" (flat, simplest ring geometry) fits in
    # small grids and appears from the earliest curriculum rungs; "semicircular"/"segmental"
    # need much more vertical room (rise scales with span) and only fit - hence only appear -
    # from mid/high rungs onward, a natural difficulty ramp gated by sample_arch_plan's own
    # grid-size filter against SIZE_LADDER, not a separately-tracked difficulty score.
    ("jack", 1, 0, 1, 2, 2, 3, 3),
    ("jack", 2, 0, 1, 2, 2, 5, 3),
    ("jack", 2, 1, 1, 3, 3, 5, 5),
    ("jack", 1, 0, 2, 3, 3, 4, 4),
    ("jack", 2, 1, 2, 2, 2, 6, 4),
    ("jack", 2, 0, 2, 3, 3, 6, 4),
    ("jack", 2, 0, 3, 2, 2, 7, 3),
    ("jack", 2, 1, 3, 3, 3, 7, 5),
    ("semicircular", 2, 1, 2, 6, 2, 6, 8),
    ("semicircular", 2, 0, 2, 7, 3, 6, 8),
    ("semicircular", 2, 0, 3, 8, 2, 7, 9),
    ("semicircular", 2, 0, 3, 9, 3, 7, 10),
    ("semicircular", 2, 1, 3, 9, 3, 7, 11),
    ("segmental", 2, 1, 2, 6, 2, 6, 8),
    ("segmental", 2, 0, 2, 7, 3, 6, 8),
    ("segmental", 2, 0, 3, 8, 2, 7, 9),
    ("segmental", 2, 0, 3, 9, 3, 7, 10),
    ("segmental", 2, 1, 3, 9, 3, 7, 11),
)


def sample_arch_plan(rng, level: int = 0) -> "FacadePlan":
    """A small facade with ONE random structural arch opening, scaled to curriculum `level` -
    the arch-curriculum analogue of blueprint.sample_spec's SIZE_LADDER. Mixed into
    BrickLayerRobotEnv.reset() at a ramping frequency (see RobotEnvConfig.arch_prob_*)
    alongside plain flat walls, so training sees increasingly larger/more varied arches
    without losing the flat-wall generalization skill. Every candidate in ARCH_PLAN_SPECS is
    pre-validated solvable; this only chooses AMONG them (gated by grid size vs. the current
    SIZE_LADDER rung), it doesn't invent new geometry that could turn out unbuildable."""
    from atrium_sim.blueprint import SIZE_LADDER

    max_m, max_c = SIZE_LADDER[min(level, len(SIZE_LADDER) - 1)]
    candidates = [c for c in ARCH_PLAN_SPECS if c[-2] <= max_m and c[-1] <= max_c]
    if not candidates:
        candidates = [min(ARCH_PLAN_SPECS, key=lambda c: c[-2] * c[-1])]
    style, col, row, n_cols, n_rows, ring_courses, grid_cols, grid_rows = (
        candidates[int(rng.integers(len(candidates)))]
    )
    o = Opening("window", col=col, row=row, n_cols=n_cols, n_rows=n_rows,
                has_lintel=False, has_sill=False, arch_style=style,
                arch_ring_courses=ring_courses)
    return FacadePlan.from_perception("curriculum", grid_cols, grid_rows, [o])


def _default_n_voussoirs(span_mm: float) -> int:
    """An odd voussoir count sized so each is roughly brick-width at mid-span - not
    masonry-optimal, just a sane default when a plan doesn't specify one explicitly."""
    n = max(3, round(span_mm / 100.0))
    return n if n % 2 == 1 else n + 1


@dataclass(frozen=True)
class FacadePanel:
    """A solid running-bond rectangle, placed at (origin_col, origin_row) in the grid."""
    spec: WallSpec
    origin_col: int
    origin_row: int
    label: str = ""


@dataclass(frozen=True)
class FacadePlan:
    image_ref: str
    grid_cols: int
    grid_rows: int
    openings: tuple[Opening, ...]
    panels: tuple[FacadePanel, ...]
    notes: str = ""

    # --- construction ---------------------------------------------------------

    @classmethod
    def from_perception(cls, image_ref: str, grid_cols: int, grid_rows: int,
                        openings: list[Opening] | tuple[Opening, ...],
                        notes: str = "") -> "FacadePlan":
        """Build a plan from the VLM's perception (grid + openings): clamp the openings
        to the grid, then tile the remaining brickwork into panels.

        A real structural arch's ring OVERSAILS onto the pier/abutment near the springing
        (BIA's abutment-width rules - the ring rests partly ON the pier, not flush at the
        opening's edge). The tiler is handed a row-banded TAPERED void for arch-styled
        openings (see `_tiling_voids`) so it never carves pier coverage into the ring's actual
        footprint in the first place - the robust alternative to generating conflicting flat
        targets and excluding them after the fact, which (discovered in-session) can leave a
        thin, unstable sliver of pier coursework next to the ring. `self.openings` keeps the
        ORIGINAL (narrower) requested opening - only the tiler sees the widened bands.

        Tiling runs in independent HORIZONTAL STRIPS, cut at every band/springing/crown
        boundary - not one `tile_facade` call over the whole grid. `tile_facade`'s greedy
        "grow up while the same columns stay free" naturally extends a narrow arch-adjacent
        band's panel past where the ring ends, since nothing above is void to stop it
        (discovered in-session: a 3-module-wide panel grew from the springing all the way to
        the grid top, permanently narrowing what should be full pier width for the whole
        wall above the arch). Cutting into strips makes every such boundary a hard one."""
        ops = tuple(_clamp_opening(o, grid_cols, grid_rows) for o in openings)
        ops = tuple(o for o in ops if o.n_cols > 0 and o.n_rows > 0)
        tiling_voids = tuple(
            v2 for o in ops for v in _tiling_voids(o)
            if (v2 := _clamp_opening(v, grid_cols, grid_rows)).n_cols > 0 and v2.n_rows > 0
        )
        cuts = sorted({0, grid_rows} | {v.row for v in tiling_voids}
                      | {v.row + v.n_rows for v in tiling_voids})
        panels: list[FacadePanel] = []
        for row0, row1 in zip(cuts, cuts[1:]):
            strip_voids = tuple(
                Opening(v.kind, v.col, max(v.row, row0) - row0, v.n_cols,
                        min(v.row + v.n_rows, row1) - max(v.row, row0),
                        v.has_lintel, v.has_sill, v.arch_style, v.arch_rise_ratio,
                        v.arch_ring_courses, v.arch_n_voussoirs)
                for v in tiling_voids if v.row < row1 and v.row + v.n_rows > row0
            )
            for p in tile_facade(grid_cols, row1 - row0, strip_voids):
                panels.append(FacadePanel(p.spec, p.origin_col, p.origin_row + row0, p.label))
        plan = cls(image_ref, grid_cols, grid_rows, ops, tuple(panels), notes)
        plan.validate()
        return plan

    # --- consumers ------------------------------------------------------------

    def blueprints(self) -> list[tuple[Blueprint, tuple[int, int]]]:
        """One running-bond Blueprint per panel, paired with its (origin_col, origin_row)
        grid offset. The renderer/oracle/audit consume these directly."""
        return [(generate_blueprint(p.spec), (p.origin_col, p.origin_row)) for p in self.panels]

    @property
    def n_bricks(self) -> int:
        return sum(generate_blueprint(p.spec).n_targets for p in self.panels)

    def hard_bodies(self) -> list[HardBody]:
        """Static (never-audited) heads/ledges/sills resolved to global mm, for the
        NON-structural styles only ("flat"'s plain lintel, "lintel_soldier"'s cosmetic fringe)
        plus every opening's sill. Each spawns once the build reaches its `trigger_course`
        ('at the exact brick level', like real construction). REAL structural arches
        ("semicircular"/"segmental"/"jack") are NOT resolved here - see `arch_regions()`: they
        are agent-placed voussoir BrickTargets on a centering, not decoration."""
        bw, bh = brick_face(BrickKind.FULL)
        out: list[HardBody] = []
        for o in self.openings:
            x0, x1 = o.col * MODULE_MM, (o.col + o.n_cols) * MODULE_MM
            top_course = o.row + o.n_rows
            if o.arch_style == "lintel_soldier":
                # the NON-structural control: a fanned SOLDIER course (near-vertical bricks,
                # tilting slightly outward from the crown) hanging just under a CEMENT lintel;
                # the lintel is the concrete framing piece AND carries the brickwork above the
                # opening - BIA: "when an arch is supported by a steel angle... the structural
                # resistance of the arch is neglected." The soldiers are cosmetic `sensor`
                # bodies (see robot_env): visible but non-colliding.
                w = x1 - x0
                cx = (x0 + x1) / 2.0
                y_top = COURSE_MM * top_course
                out.append(HardBody("cement",
                    ((x0, y_top - COURSE_MM), (x1, y_top - COURSE_MM), (x1, y_top), (x0, y_top)),
                    trigger_course=max(0, top_course - 1)))
                nv = max(3, round(w / bh))
                max_tilt = math.radians(16.0)
                y_sold = (y_top - COURSE_MM) - bw / 2.0
                for k in range(nv):
                    u = 2.0 * (k + 0.5) / nv - 1.0
                    xi = cx + u * (w / 2.0 - bh / 2.0)
                    th = math.pi / 2.0 - u * max_tilt
                    c, s = math.cos(th), math.sin(th)
                    corners = tuple(
                        (xi + dx * c - dy * s, y_sold + dx * s + dy * c)
                        for dx, dy in ((-bw / 2, -bh / 2), (bw / 2, -bh / 2),
                                       (bw / 2, bh / 2), (-bw / 2, bh / 2)))
                    course = round((y_sold - 30.0) / COURSE_MM)
                    out.append(HardBody("voussoir", corners, trigger_course=max(0, course - 1)))
            elif o.arch_style == "flat" and o.has_lintel and top_course < self.grid_rows:
                top = COURSE_MM * top_course
                verts = ((x0, top - COURSE_MM), (x1, top - COURSE_MM), (x1, top), (x0, top))
                out.append(HardBody("lintel", verts, trigger_course=top_course - 1))
            if o.has_sill and o.row > 0:
                # a thin ledge sitting ON TOP of the course below the opening (inset from the
                # piers), in the opening's void base - not overlapping any brick (which flings it)
                base = COURSE_MM * o.row
                verts = ((x0 + 12, base), (x1 - 12, base),
                         (x1 - 12, base + COURSE_MM * 0.5), (x0 + 12, base + COURSE_MM * 0.5))
                out.append(HardBody("sill", verts, trigger_course=o.row - 1))
        return out

    def arch_regions(self) -> tuple["ArchRegion", ...]:
        """Every REAL structural arch (arch_style in semicircular/segmental/jack), resolved to
        global mm - the buildable counterpart to `hard_bodies()`'s decorative/flat elements.
        The springing sits high enough in the opening's reserved void that the ring's rise fits
        entirely below the void's own top (== where the tiler already resumes flat brickwork);
        below the springing, within the opening's column span, stays genuinely empty (the
        window/door's clear opening)."""
        out = []
        for i, o in enumerate(self.openings):
            resolved = _resolve_arch(o)
            if resolved is None:
                continue
            spec, spring_course, top_course = resolved
            void_x0, void_x1 = _arch_void_bounds_mm(o, spec)
            out.append(ArchRegion(
                opening_index=i, spec=spec,
                origin_x=(o.col + o.n_cols / 2.0) * MODULE_MM,
                spring_y=COURSE_MM * spring_course,
                springing_course=spring_course, crown_course=top_course,
                void_x0=void_x0, void_x1=void_x1,
            ))
        return tuple(out)

    # --- validation -----------------------------------------------------------

    def validate(self) -> "FacadePlan":
        """Panels + openings are in-grid, panels don't overlap each other or openings."""
        if self.grid_cols <= 0 or self.grid_rows <= 0:
            raise ValueError(f"degenerate grid {self.grid_cols}x{self.grid_rows}")
        for o in self.openings:
            if not _in_grid(o.col, o.row, o.n_cols, o.n_rows, self.grid_cols, self.grid_rows):
                raise ValueError(f"opening {o} out of grid")
        boxes = [(p.origin_col, p.origin_row, p.spec.n_modules, p.spec.n_courses)
                 for p in self.panels]
        for i, b in enumerate(boxes):
            if not _in_grid(*b, self.grid_cols, self.grid_rows):
                raise ValueError(f"panel {self.panels[i]} out of grid")
            for o in self.openings:
                if _overlap(b, (o.col, o.row, o.n_cols, o.n_rows)):
                    raise ValueError(f"panel {self.panels[i]} overlaps opening {o}")
            for j in range(i + 1, len(boxes)):
                if _overlap(b, boxes[j]):
                    raise ValueError(f"panels {self.panels[i]} and {self.panels[j]} overlap")
        return self

    # --- json (the VLM output contract, round-trippable) ----------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            "image_ref": self.image_ref,
            "grid_cols": self.grid_cols, "grid_rows": self.grid_rows,
            "openings": [o.__dict__ for o in self.openings],
            "panels": [{"n_modules": p.spec.n_modules, "n_courses": p.spec.n_courses,
                        "origin_col": p.origin_col, "origin_row": p.origin_row,
                        "label": p.label} for p in self.panels],
            "notes": self.notes,
        }, indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "FacadePlan":
        d = json.loads(s)
        return cls(
            d["image_ref"], d["grid_cols"], d["grid_rows"],
            tuple(Opening(**o) for o in d["openings"]),
            tuple(FacadePanel(WallSpec(p["n_modules"], p["n_courses"]),
                              p["origin_col"], p["origin_row"], p.get("label", ""))
                  for p in d["panels"]),
            d.get("notes", ""),
        )


# --- deterministic tiler ------------------------------------------------------

def tile_facade(grid_cols: int, grid_rows: int,
                openings: tuple[Opening, ...]) -> tuple[FacadePanel, ...]:
    """Partition the brick cells (grid minus openings) into non-overlapping rectangular
    running-bond panels. Greedy maximal-rectangle: scan bottom-left to top-right; at each
    unclaimed brick cell grow right while brick, then grow up while every column stays
    brick, claim that rectangle as a panel. Valid + deterministic (not masonry-optimal)."""
    void = [[False] * grid_cols for _ in range(grid_rows)]
    for o in openings:
        for r in range(o.row, min(o.row + o.n_rows, grid_rows)):
            for c in range(o.col, min(o.col + o.n_cols, grid_cols)):
                void[r][c] = True
    claimed = [[False] * grid_cols for _ in range(grid_rows)]
    panels: list[FacadePanel] = []
    for r0 in range(grid_rows):
        for c0 in range(grid_cols):
            if void[r0][c0] or claimed[r0][c0]:
                continue
            c1 = c0
            while c1 < grid_cols and not void[r0][c1] and not claimed[r0][c1]:
                c1 += 1
            r1 = r0
            while r1 < grid_rows and all(
                not void[r1][c] and not claimed[r1][c] for c in range(c0, c1)
            ):
                r1 += 1
            for r in range(r0, r1):
                for c in range(c0, c1):
                    claimed[r][c] = True
            panels.append(FacadePanel(WallSpec(c1 - c0, r1 - r0), c0, r0))
    return tuple(panels)


# --- helpers ------------------------------------------------------------------

def _resolve_arch(o: "Opening") -> tuple[ArchSpec, int, int] | None:
    """(ArchSpec, springing_course, crown_course) for a structural-arch-styled opening, or
    None otherwise. Shared by `FacadePlan.arch_regions()` and `_tiling_voids()` (which needs
    the same geometry BEFORE a FacadePlan exists, while tiling) so the two can never drift
    out of sync with each other."""
    if o.arch_style not in STRUCTURAL_ARCH_STYLES:
        return None
    span = o.n_cols * MODULE_MM
    top_course = o.row + o.n_rows
    ring_depth = COURSE_MM * o.arch_ring_courses
    if o.arch_style == "jack":
        rise = 1.0   # BIA: a jack arch's camber (~1/96 of span) is cosmetic, not modeled
        needed_courses = o.arch_ring_courses
    else:
        rise = span * o.arch_rise_ratio
        # the ring's TRUE vertical extent above the spring line is rise + ring_depth (the
        # crown voussoir's own extrados apex), not rise alone - using rise alone under-
        # reserved room and let the crown voussoir physically overlap where flat coursing
        # resumes (discovered in-session: ~196mm of overlap on a 660mm-span semicircular
        # arch with a 4-course ring - not a small margin).
        needed_courses = max(1, math.ceil((rise + ring_depth) / COURSE_MM))
    spring_course = max(o.row, top_course - needed_courses)
    n_vous = o.arch_n_voussoirs or _default_n_voussoirs(span)
    spec = ArchSpec(kind=o.arch_style, span_mm=span, rise_mm=rise,
                     ring_depth_mm=ring_depth, n_voussoirs=n_vous)
    return spec, spring_course, top_course


def _void_half_bounds(spec: ArchSpec) -> tuple[float, float]:
    """The ring's WORST-CASE (widest-row) reach, as (lo, hi) offsets from the arch centerline
    in its own local frame - the tiler's own void bound for an arch-styled opening (see
    `_tiling_voids`), and the same bound `ArchRegion.crown_packing_hard_body`/
    `spandrel_hard_bodies` fill to - shared so the tiled void and the packing that closes its
    gaps against the ring's true (narrower, per-row) shape can never drift out of sync."""
    from atrium_sim.arch import ring_row_spans

    spans = ring_row_spans(spec)
    lo, hi = -spec.span_mm / 2.0, spec.span_mm / 2.0
    for intervals in spans.values():
        for local_lo, local_hi in intervals:
            lo = min(lo, local_lo)
            hi = max(hi, local_hi)
    return lo, hi


def _arch_void_bounds_mm(o: Opening, spec: ArchSpec) -> tuple[float, float]:
    """The tiler's own QUANTIZED (whole-module) void bounds, in GLOBAL mm, for opening `o`'s
    arch. Shared by `_tiling_voids` (which excludes flat pier targets up to exactly this bound)
    and `FacadePlan.arch_regions()` (which needs the SAME bound so `ArchRegion`'s crown/spandrel
    packing fills exactly the gap the tiler leaves - not the ring's own narrower, unquantized
    reach, which (discovered in-session) falls short of the tiler's module-rounded edge by up
    to a whole column and leaves an unsupported strip along that edge at every course)."""
    origin_x = (o.col + o.n_cols / 2.0) * MODULE_MM
    local_lo, local_hi = _void_half_bounds(spec)
    lo = min(o.col * MODULE_MM, origin_x + local_lo)
    hi = max((o.col + o.n_cols) * MODULE_MM, origin_x + local_hi)
    col0 = int(math.floor(lo / MODULE_MM))
    col1 = int(math.ceil(hi / MODULE_MM))
    return col0 * MODULE_MM, col1 * MODULE_MM


def _tiling_voids(o: Opening) -> list[Opening]:
    """The opening's footprint AS FAR AS THE TILER IS CONCERNED: for a real structural arch
    style, ONE rectangle sized to the ring's WORST-CASE (widest) reach across every row from
    the springing to the crown (see atrium_sim.arch.ring_row_spans).

    This is deliberately NOT tapered to the ring's actual (narrower, near the crown) reach -
    two tapering attempts were tried and reverted in-session: per-row voids broke running bond
    (a panel only 1 course tall can't represent a true odd/half-brick-ended course), and
    multi-course BANDS avoided that but still left an unsupported overhang wherever a band's
    reach happened to widen versus the band below (a real ring's cross-section isn't strictly
    monotonic course to course). A single uniform void sidesteps both failure modes; the
    resulting over-wide gap between the ring's true (curved, non-course-aligned) surface and
    this rectangle's edges is filled with static SPANDREL PACKING instead (see
    `ArchRegion.spandrel_hard_bodies` - real masonry fills exactly this space between an arch's
    extrados and the surrounding rectangular coursing, so this isn't a workaround, it's the
    normal construction detail).

    Rows below the springing (the genuine clear window/door void) use the opening's own plain
    column range unwidened. Non-arch styles return `[o]` unchanged."""
    resolved = _resolve_arch(o)
    if resolved is None:
        return [o]
    spec, spring_course, crown_course = resolved
    out = []
    if spring_course > o.row:
        out.append(Opening(o.kind, o.col, o.row, o.n_cols, spring_course - o.row,
                            o.has_lintel, o.has_sill, o.arch_style, o.arch_rise_ratio,
                            o.arch_ring_courses, o.arch_n_voussoirs))
    x0, x1 = _arch_void_bounds_mm(o, spec)
    col0 = int(round(x0 / MODULE_MM))
    col1 = int(round(x1 / MODULE_MM))
    out.append(Opening(o.kind, col0, spring_course, col1 - col0, crown_course - spring_course,
                        o.has_lintel, o.has_sill, o.arch_style, o.arch_rise_ratio,
                        o.arch_ring_courses, o.arch_n_voussoirs))
    return out


def _clamp_opening(o: Opening, cols: int, rows: int) -> Opening:
    col = max(0, min(o.col, cols))
    row = max(0, min(o.row, rows))
    return Opening(o.kind, col, row, max(0, min(o.n_cols, cols - col)),
                   max(0, min(o.n_rows, rows - row)), o.has_lintel, o.has_sill,
                   o.arch_style, o.arch_rise_ratio, o.arch_ring_courses, o.arch_n_voussoirs)


def _in_grid(col: int, row: int, nc: int, nr: int, cols: int, rows: int) -> bool:
    return col >= 0 and row >= 0 and nc >= 0 and nr >= 0 and col + nc <= cols and row + nr <= rows


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
