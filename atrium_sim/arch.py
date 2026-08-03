"""Real structural arches over openings: voussoir geometry, centering, build order, audit.

Companion to blueprint.py/facade.py at the same import-light tier (no torch/gym/pymunk).

Physically de-risked before writing this module (in-session spike, same PyMunk config as
physics.py - friction 0.9, collision_slop 0.1, 30 solver iterations): a ring of TAPERED wedge
voussoirs (radial, face-to-face joints - not rectangular bricks resting on a curve) genuinely
stands, needs a temporary centering until the ring closes at the keystone, and collapses if
built in the wrong order or left incomplete when struck. This replaces the earlier design
where arches were decorative `sensor` bodies (kept as facade.py's "lintel_soldier" style, the
non-structural control - BIA itself describes this as standard modern practice: "when an arch
is supported by a steel angle... the structural resistance of the arch is neglected").

Two geometrically distinct families:
- "semicircular" / "segmental": a circular arc through both springings and the crown. Radially
  uniform, so every voussoir shares one canonical (unrotated) wedge shape - only its angular
  position differs.
- "jack": BIA's "flat arch with zero or little rise" - a genuinely FLAT intrados/extrados, with
  joints fanned from a striking point below the spring line (BIA TN 31A eq. 4: skewback angle
  gamma = atan(span_ft/8), independent of ring depth). Each voussoir has its own trapezoid shape
  (the flat top/bottom means adjacent joints are not parallel), unlike the arc kinds.

Convention (matches BrickTarget's existing flat-brick convention exactly, just generalized):
a target's `wedge_verts` is the wedge's shape in a CANONICAL frame at theta=0, centroid-
relative. `physics.spawn_brick(..., theta=target.theta + agent_error, wedge_verts=...)` applies
the intended rotation (plus the agent's small nudge) the same way it already applies `theta` to
flat bricks. For arc kinds this canonical shape is identical for every voussoir (curvature is
uniform); for jack it differs per voussoir but the same spawn-time convention still applies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ARCH_KINDS = ("semicircular", "segmental", "jack")

ARCH_JOINT_GAP_MM = 0.8
# Effective joint clearance between voussoirs. Validated in-session: 0.2mm gap -> 4.3mm crown
# settlement on striking a closed 11-voussoir ring; 4mm gap -> 35.9mm. A smaller gap (tried
# 0.2/0.4mm first) occasionally left the keystone's final gap too tight for the spawn probe
# after 8 prior voussoirs' accumulated settling (probe-exhaustion - only ever seen on the
# 9-voussoir jack case, and only intermittently). 0.8mm gives the last insertion reliable
# margin while keeping settlement modest; still within BIA's gauged-to-uncut joint-width
# range (1/8-3/4in / 3-19mm).

LEVEL_PAD_MM = 3.0
# Thickness of the static bearing pad substituted for a degenerate (zero-area) skewback wedge
# (semicircular) or a jack's flat intrados - a fresh, precisely-level reference surface for the
# springer, matching a real bedding/DPC course, rather than relying on the actual pier top's
# accumulated settle-slop being exactly flat.

MIN_THICKNESS_RATIO = 0.15
# Safety margin above the Couplet/Heyman/Milankovitch minimum thickness ratio for a
# semicircular arch (t/R ~ 0.1075, the exact value below which NO thrust line fits in the ring
# at all). Validated in-session: t/R=0.182 stands cleanly, t/R=0.106 (below the true limit)
# collapses outright (571mm drift, 230deg tilt). Not meaningful for "jack" (no single radius).

VOUSSOIR_TOL_MM = 5.0
VOUSSOIR_SIGMA_MM = 10.0
VOUSSOIR_TOL_DEG = 3.0
VOUSSOIR_SIGMA_DEG = 6.0
# Placement-quality tolerance for an individual voussoir, BEFORE the strike (graded, like a
# flat brick's tol_mm/tol_deg but wider - physically correct, not a concession: real arch
# joints run 3-19mm depending on style, vs a flat course's mortar bed. Grounded in-session: the
# closed ring is fully stable (no collapse) through +-12mm tangential slide / +-4deg off-radial
# tilt per voussoir; it only degrades gracefully (more settlement) beyond that.

SURVIVAL_DRIFT_MM = 20.0
SURVIVAL_TILT_DEG = 10.0
# Post-strike survival thresholds. Grounded in-session: a correctly closed, symmetrically-built
# ring settles 4-16mm on striking even under placement error; a ring missing one voussoir, or
# built in the wrong order, moves 100mm+ and topples. 20mm/10deg sits well above real
# settlement and well below real collapse - a clean separator, not a tight tolerance.


@dataclass(frozen=True)
class ArchSpec:
    """A real structural masonry arch spanning an opening, in LOCAL coordinates: x=0 at the
    arch centerline, y=0 at the spring line (top of the abutment piers)."""

    kind: str
    span_mm: float
    rise_mm: float          # ignored for "jack" (its flatness is a BIA-documented ~1/96 camber,
                            # cosmetic - not modeled as a curve)
    ring_depth_mm: float    # radial ring thickness (arc kinds) / vertical depth (jack)
    n_voussoirs: int        # must be odd - BIA: a joint at the crown is the weakest point;
                            # an odd count with a keystone moves the first potential crack away
                            # from midspan

    def __post_init__(self) -> None:
        if self.kind not in ARCH_KINDS:
            raise ValueError(f"unknown arch kind {self.kind!r}")
        if self.n_voussoirs < 3 or self.n_voussoirs % 2 == 0:
            raise ValueError(f"n_voussoirs must be odd and >= 3, got {self.n_voussoirs}")
        if self.kind != "jack" and not (0.0 < self.rise_mm < self.span_mm / 2.0 + 1e-6):
            raise ValueError(f"rise_mm {self.rise_mm} out of range for span {self.span_mm}")
        if self.ring_depth_mm <= 0.0:
            raise ValueError("ring_depth_mm must be positive")

    # --- circular-arc geometry (semicircular / segmental) --------------------------------

    @property
    def radius_mm(self) -> float:
        """R = rise/2 + span^2/(8*rise) - the circle through both springings and the crown
        (BIA/standard arch geometry; semicircular is the special case rise = span/2 -> R =
        span/2)."""
        return self.rise_mm / 2.0 + (self.span_mm ** 2) / (8.0 * self.rise_mm)

    @property
    def half_angle_rad(self) -> float:
        """Half the angle of embrace, from the vertical crown axis to each springing:
        sin(half_angle) = (span/2) / R."""
        return math.asin(min(1.0, (self.span_mm / 2.0) / self.radius_mm))

    @property
    def centre_y_mm(self) -> float:
        """The arc's centre of curvature, in the same local frame (y=0 at spring line).
        Below the spring line unless rise == R (semicircular, centre AT the spring line)."""
        return self.rise_mm - self.radius_mm

    # --- jack-arch geometry ----------------------------------------------------------------

    @property
    def jack_skewback_rad(self) -> float:
        """Skewback angle from vertical: tan(gamma) = span_ft / 8 (BIA TN 31A eq. 4),
        independent of ring depth."""
        span_ft = self.span_mm / 304.8
        return math.atan(span_ft / 8.0)

    # --- validation --------------------------------------------------------------------

    def min_thickness_ok(self) -> bool:
        """Couplet/Heyman/Milankovitch minimum-thickness check, with a safety margin. Always
        True for "jack" (a flat arch has no single-radius thickness ratio to check)."""
        if self.kind == "jack":
            return True
        return (self.ring_depth_mm / self.radius_mm) >= MIN_THICKNESS_RATIO


@dataclass(frozen=True)
class Wedge:
    """One voussoir's target pose + shape, in the SAME convention as BrickTarget: `verts` is
    the canonical (theta=0) shape, centroid-relative; `x, y, theta` is the intended world pose
    (in the arch's local frame - callers offset into global coordinates)."""

    index: int      # 0..n_voussoirs-1, left to right (0 = left springer, geometric order -
                    # NOT build order; see build_order())
    x: float
    y: float
    theta: float
    verts: tuple[tuple[float, float], ...]


def _rotate(dx: float, dy: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return dx * c - dy * s, dx * s + dy * c


def _centroid(verts: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    return sum(p[0] for p in verts) / len(verts), sum(p[1] for p in verts) / len(verts)


def _polygon_area(verts: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * abs(sum(
        verts[i][0] * verts[(i + 1) % len(verts)][1] - verts[(i + 1) % len(verts)][0] * verts[i][1]
        for i in range(len(verts))
    ))


def wedge_verts_mass_kg(wedge_verts: tuple[tuple[float, float], ...]) -> float:
    """A voussoir's mass, scaled from BRICK_MASS_KG by its polygon area relative to a
    standard full brick's envelope area - NOT a fixed per-brick mass, since wedges vary in
    size with arch geometry (this was the root cause of an earlier validation failure: a
    hardcoded mass gave a segmental ring ~13x too little mass relative to its true size,
    destabilizing it under the same contact solver that handles a correctly-massed ring
    fine). Takes raw verts (what BrickTarget.wedge_verts carries) rather than a Wedge, so
    callers scoring a placed target don't need to re-derive one."""
    from atrium_sim.constants import BRICK_MASS_KG, FULL_ENVELOPE

    return BRICK_MASS_KG * _polygon_area(wedge_verts) / (FULL_ENVELOPE[0] * FULL_ENVELOPE[1])


def wedge_mass_kg(wedge: Wedge) -> float:
    return wedge_verts_mass_kg(wedge.verts)


def _finalize(true_verts: tuple[tuple[float, float], ...], theta: float, index: int) -> Wedge:
    """Given a wedge's TRUE final shape (already correctly positioned/oriented, in the arch's
    local frame) and the `theta` physics should apply at spawn time, de-rotate the shape by
    `-theta` around its own centroid so `wedge_verts` is canonical (matches BrickTarget's
    existing theta=0-canonical / spawn-applies-theta convention for flat bricks)."""
    cx, cy = _centroid(true_verts)
    local = tuple(_rotate(px - cx, py - cy, -theta) for px, py in true_verts)
    return Wedge(index=index, x=cx, y=cy, theta=theta, verts=local)


def _arc_wedge_quad(ri: float, ro: float, dphi: float, gap_mm: float
                     ) -> tuple[tuple[float, float], ...]:
    """The ONE canonical wedge shape shared by every voussoir in an arc-type ring (curvature
    is uniform, so only angular position - not shape - varies): a trapezoid between radii
    ri..ro over angular width dphi, centred on the vertical (phi=0) axis, shrunk by gap_mm/2 of
    arc at mid-radius so adjacent voussoirs' Minkowski-inflated faces touch without
    overlapping. Built via `_rotate` (standard CCW convention) so it stays sign-consistent
    with how `_arc_wedges` later rotates each voussoir's centroid by the same convention; CCW
    winding overall (required by pymunk.Poly for a positive-area convex shape)."""
    rm = 0.5 * (ri + ro)
    half = dphi / 2.0 - (gap_mm / 2.0) / rm
    return (
        _rotate(0.0, ri, half),
        _rotate(0.0, ri, -half),
        _rotate(0.0, ro, -half),
        _rotate(0.0, ro, half),
    )


def _arc_wedges(spec: ArchSpec) -> tuple[Wedge, ...]:
    ri = spec.radius_mm
    ro = ri + spec.ring_depth_mm
    cy = spec.centre_y_mm
    ha = spec.half_angle_rad
    dphi = 2.0 * ha / spec.n_voussoirs
    canonical = _arc_wedge_quad(ri, ro, dphi, ARCH_JOINT_GAP_MM)
    cx0, cy0 = _centroid(canonical)
    out = []
    for k in range(spec.n_voussoirs):
        # `_rotate`'s standard-CCW convention maps a POSITIVE angle to NEGATIVE x (for a point
        # starting on the +y axis) - so index 0 (intended: left springer) must get the LARGEST
        # phi, not the smallest, to land on the geometrically-left (negative-x) side.
        phi = ha - dphi * (k + 0.5)
        # world centroid: rotate the canonical shape's own centroid (which sits on the phi=0
        # axis, at local (0, ~rm) relative to the arc centre) by phi around the arc centre,
        # using the SAME `_rotate` convention the canonical shape itself was built with.
        wx, wy = _rotate(cx0, cy0, phi)
        out.append(Wedge(index=k, x=wx, y=wy + cy, theta=phi,
                          verts=tuple((px - cx0, py - cy0) for px, py in canonical)))
    return tuple(out)


def _jack_wedges(spec: ArchSpec) -> tuple[Wedge, ...]:
    """Flat intrados/extrados; joints are straight radial lines through a striking point
    below the spring line. Each voussoir's exact trapezoid differs (flat top/bottom means the
    left/right joints are not parallel), so - unlike the arc kinds - there is no single shared
    canonical shape; each is de-rotated individually by its own mean joint angle."""
    gamma = spec.jack_skewback_rad
    d = spec.ring_depth_mm
    strike_depth = (spec.span_mm / 2.0) / max(math.tan(gamma), 1e-9)
    dphi = 2.0 * gamma / spec.n_voussoirs
    joint_angles = [-gamma + dphi * k for k in range(spec.n_voussoirs + 1)]
    gap = ARCH_JOINT_GAP_MM / 2.0
    out = []
    for k in range(spec.n_voussoirs):
        lo, hi = joint_angles[k], joint_angles[k + 1]
        xi_lo = strike_depth * math.tan(lo)
        xi_hi = strike_depth * math.tan(hi)
        xo_lo = (strike_depth + d) * math.tan(lo)
        xo_hi = (strike_depth + d) * math.tan(hi)
        # shrink each flat edge inward by gap/2 per side (tangential shrink, mirrors the arc
        # case's arc-length shrink) so adjacent voussoirs don't overlap once inflated.
        fi = gap / max(xi_hi - xi_lo, 1e-6)
        fo = gap / max(xo_hi - xo_lo, 1e-6)
        xi_lo2 = xi_lo + (xi_hi - xi_lo) * fi
        xi_hi2 = xi_hi - (xi_hi - xi_lo) * fi
        xo_lo2 = xo_lo + (xo_hi - xo_lo) * fo
        xo_hi2 = xo_hi - (xo_hi - xo_lo) * fo
        true_verts = ((xi_lo2, 0.0), (xi_hi2, 0.0), (xo_hi2, d), (xo_lo2, d))
        mean_angle = 0.5 * (lo + hi)
        out.append(_finalize(true_verts, mean_angle, k))
    return tuple(out)


def arch_wedges(spec: ArchSpec) -> tuple[Wedge, ...]:
    """Every voussoir's local target pose + canonical shape, left to right (geometric index
    order - NOT build order; see build_order())."""
    return _jack_wedges(spec) if spec.kind == "jack" else _arc_wedges(spec)


def _row_band_x_range(world_verts: list[tuple[float, float]], y0: float, y1: float
                       ) -> tuple[float, float] | None:
    """min/max x of a (convex) polygon's intersection with the horizontal band [y0, y1), from
    its own vertices that fall in-band plus any edge crossing of y0/y1. None if the polygon
    doesn't reach this band at all."""
    n = len(world_verts)
    xs: list[float] = []
    for i in range(n):
        xa, ya = world_verts[i]
        xb, yb = world_verts[(i + 1) % n]
        if y0 <= ya <= y1:
            xs.append(xa)
        for yline in (y0, y1):
            if (ya - yline) * (yb - yline) < 0:  # edge straddles this line
                t = (yline - ya) / (yb - ya)
                xs.append(xa + t * (xb - xa))
    return (min(xs), max(xs)) if xs else None


def ring_row_spans(spec: ArchSpec, course_mm: float | None = None
                    ) -> dict[int, tuple[tuple[float, float], ...]]:
    """For each course-row-offset from the spring line (row 0 = local y in [0, course_mm)) that
    the ring's geometry reaches, the DISJOINT x-intervals - in the arch's own LOCAL frame -
    occupied by any voussoir whose (axis-aligned) bounding box overlaps that row.

    Why this exists: a real arch ring generally OVERSAILS onto the abutment/pier near the
    springing (the springer voussoir's outer edge rests partly ON the pier, not flush at the
    opening's edge - exactly what BIA's abutment-width rules are about). Discovered in-session
    the hard way: with a plain rectangular opening void, ordinary flat pier coursework was
    still being generated at the same (course, x) the ring's springers physically occupy, and
    the two collided the instant both existed - not just at the springing course, but at every
    course the ring's base still reaches into pier territory. blueprint.py/facade.py use this
    to exclude flat targets from exactly the rows/columns the ring will occupy.

    Returns per-row DISJOINT intervals, not a single collapsed (min, max): a first
    implementation collapsed to one span and, for a semicircular ring, silently merged the
    left springer's interval with the mirror-image right springer's interval into one giant
    "everything in between" exclusion - wiping out legitimate pier coursework far from either
    springer that the ring never actually reaches.

    Each row's interval is the wedge's ACTUAL cross-section at that row band (clipped polygon
    intersection), not its whole axis-aligned bounding box - a tilted wedge's bbox is a poor
    over-approximation of its footprint at any single row (e.g. a springer near 90 degrees has
    a ~250mm-tall bbox but is only a sliver-wide at its lowest row). Using the whole bbox for
    every row it merely TOUCHES excluded enough of a real pier's own legitimate coursing, across
    several rows at once, to leave an unstable isolated sliver next to the void - discovered
    when a full facade build (not just the arch in isolation) finally exercised this path."""
    from atrium_sim.constants import COURSE_MM as _COURSE_MM

    cm = course_mm or _COURSE_MM
    raw: dict[int, list[tuple[float, float]]] = {}
    for w in arch_wedges(spec):
        world = [(w.x + _rotate(lx, ly, w.theta)[0], w.y + _rotate(lx, ly, w.theta)[1])
                 for lx, ly in w.verts]
        ys = [p[1] for p in world]
        lo_row = max(0, int(min(ys) // cm))
        # a wedge's top edge landing EXACTLY on a course boundary (e.g. ring_depth an exact
        # multiple of COURSE_MM) would otherwise register a degenerate, zero-height sliver in
        # the row ABOVE the ring's true extent - discovered in-session: this spuriously excluded
        # flat crown-course targets the ring never actually occupies, stalling the wall exactly
        # one course above a fully-closed, fully-survived arch. A tiny epsilon keeps an exact
        # touch from spilling into the next row.
        hi_row = int((max(ys) - 1e-6) // cm)
        for r in range(lo_row, hi_row + 1):
            band = _row_band_x_range(world, r * cm, (r + 1) * cm)
            if band is not None:
                raw.setdefault(r, []).append(band)
    spans: dict[int, tuple[tuple[float, float], ...]] = {}
    for r, intervals in raw.items():
        intervals.sort()
        merged = [intervals[0]]
        for lo, hi in intervals[1:]:
            if lo <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        spans[r] = tuple(merged)
    return spans


def build_order(n_voussoirs: int) -> tuple[int, ...]:
    """Springings-to-keystone symmetric sequence: the ONLY build order validated to survive
    striking in-session. An asymmetric order (e.g. strict left-to-right) collapses the ring on
    striking even when every voussoir ends up in the identical final position - construction
    SEQUENCE, not just final geometry, determines whether the ring survives."""
    mid = (n_voussoirs - 1) / 2.0
    return tuple(sorted(range(n_voussoirs), key=lambda k: -abs(k - mid)))


def abutment_wedge_verts(spec: ArchSpec, side: int) -> tuple[tuple[float, float], ...]:
    """A solid, PERMANENT (never struck) filler block bridging the flat pier top (y=0) and
    the arch's raked skewback plane, for `side` = -1 (left) or +1 (right). Local coords.

    Discovered necessary in-session: for any arch whose skewback isn't horizontal (segmental;
    also jack, though its own flat-trapezoid geometry already sits flush - see below), the
    first (springer) voussoir's outer face meets a plain flat pier top at a single edge, not
    flush - an unstable line contact that let the springer rotate ~54 degrees off target and
    collapse the ring (349mm drift) even with a correct ring and correct build order. Adding
    this wedge (giving the springer a flush face-to-face bed, exactly BIA's "the springing
    brick should be cut... to allow vertical alignment with the brick beneath") fixed it
    outright (349mm -> 1.4mm drift). Real masonry practice: skewbacks are cut and laid FIRST,
    before the centering goes in - this is that cut, made explicit as geometry.

    Semicircular is the special case where the skewback IS horizontal (half_angle = 90deg) -
    the triangle above degenerates to a zero-area sliver there (all three points collapse onto
    the spring line). Emitting a degenerate `pymunk.Poly` from that is NOT harmless as first
    assumed: it corrupted the springer's contact outright (settled ~60mm - almost exactly one
    course - above its target, discovered when a full facade build finally exercised the
    semicircular case end-to-end, unlike the direct-physics validation that happened not to
    trigger it, and only ever built 4 idealized pier courses). Below the min-area threshold,
    a THIN LEVEL BEARING PAD is emitted instead of nothing at all: on a full facade build (10+
    accumulated courses of settle-slop below the springing, vs. an idealized 4-course pier),
    the real pier top isn't perfectly flat by the time it reaches the springing course, and a
    semicircular springer's horizontal outer face - which needs a precisely level bed, not just
    A bed - was found to still destabilize resting directly on it. The pad gives it a fresh,
    guaranteed-level reference surface instead, exactly the function of a masonry bedding/DPC
    course. Jack's intrados is flat at y=0 too, but not necessarily flush with real pier
    settling either - it gets the same pad treatment.
    """
    half_span = spec.span_mm / 2.0
    x_edge = half_span if side > 0 else -half_span
    if spec.kind == "jack":
        oversail = spec.ring_depth_mm
        x_far = x_edge + side * oversail
        lo, hi = sorted((x_edge, x_far))
        return ((lo, 0.0), (hi, 0.0), (hi, LEVEL_PAD_MM), (lo, LEVEL_PAD_MM))
    ri, ro = spec.radius_mm, spec.radius_mm + spec.ring_depth_mm
    cy = spec.centre_y_mm
    angle = spec.half_angle_rad if side < 0 else -spec.half_angle_rad
    ix, iy = _rotate(0.0, ri, angle)
    ox, oy = _rotate(0.0, ro, angle)
    verts = ((ix, iy + cy), (ox, oy + cy), (ox, 0.0))
    area = 0.5 * sum(
        verts[i][0] * verts[(i + 1) % 3][1] - verts[(i + 1) % 3][0] * verts[i][1]
        for i in range(3)
    )
    if abs(area) < 1.0:   # degenerate (e.g. semicircular's horizontal skewback) - level pad instead
        oversail = ro - half_span
        x_far = x_edge + side * oversail
        lo, hi = sorted((x_edge, x_far))
        return ((lo, 0.0), (hi, 0.0), (hi, LEVEL_PAD_MM), (lo, LEVEL_PAD_MM))
    return verts if area > 0 else tuple(reversed(verts))


def centering_polygon(spec: ArchSpec, half_width_pad_mm: float = 0.0
                       ) -> tuple[tuple[float, float], ...]:
    """A convex polygon (arch-local coords, spring line at y=0) filling the void under the
    intrados - the temporary formwork the ring rests on until it closes at the keystone and is
    struck. Arc kinds: an N-gon under the arc. Jack: a flat pad under the flat intrados."""
    half_span = spec.span_mm / 2.0 + half_width_pad_mm
    if spec.kind == "jack":
        return ((-half_span, 0.0), (half_span, 0.0), (half_span, -1.0), (-half_span, -1.0))
    ri = spec.radius_mm
    cy = spec.centre_y_mm
    ha = spec.half_angle_rad
    n = 48
    arc = [
        (ri * math.sin(-ha + 2 * ha * i / n), cy + ri * math.cos(-ha + 2 * ha * i / n))
        for i in range(n + 1)
    ]
    return tuple([(-half_span, 0.0), *arc, (half_span, 0.0)])


# --- reporting helpers (pure; robot_env assembles these into per-episode state) -------------


def voussoir_quality(d_mm: float, dtheta_rad: float) -> float:
    """Same multiplicative plateau-Gaussian shape as reward.brick_quality, but with the wider,
    physically-appropriate voussoir tolerance (see module docstring)."""
    s_pos = _plateau_gauss(d_mm, VOUSSOIR_TOL_MM, VOUSSOIR_SIGMA_MM)
    s_ang = _plateau_gauss(abs(math.degrees(dtheta_rad)), VOUSSOIR_TOL_DEG, VOUSSOIR_SIGMA_DEG)
    return s_pos * s_ang


def _plateau_gauss(v: float, tol: float, sigma: float) -> float:
    if v <= tol:
        return 1.0
    return math.exp(-(((v - tol) / sigma) ** 2))


def ring_drift(before: dict[int, tuple[float, float, float]],
                after: dict[int, tuple[float, float, float]]) -> tuple[float, float]:
    """Max centroid movement (mm) and max angle change (deg) across a set of (x, y, theta)
    snapshots keyed by target id, e.g. across the strike-settle."""
    max_pos = 0.0
    max_ang = 0.0
    for tid, (x0, y0, th0) in before.items():
        if tid not in after:
            continue
        x1, y1, th1 = after[tid]
        max_pos = max(max_pos, math.hypot(x1 - x0, y1 - y0))
        max_ang = max(max_ang, abs(math.degrees(th1 - th0)))
    return max_pos, max_ang


def survived(max_drift_mm: float, max_tilt_deg: float) -> bool:
    return max_drift_mm < SURVIVAL_DRIFT_MM and max_tilt_deg < SURVIVAL_TILT_DEG


def middle_third_ok(eccentricity_mm: float, ring_depth_mm: float) -> bool:
    """BIA's middle-third rule: the thrust line must stay within the middle third of the ring
    depth, i.e. |e| <= d/6."""
    return abs(eccentricity_mm) <= ring_depth_mm / 6.0
