"""Structured state observation: slot tensor + globals, flattened to (538,).

Per-slot features are read from the SAME AuditReport the reward uses, so the
observation and the reward can never disagree about wall state. The tensor is
padded to the largest wall any suite generates (C_MAX x S_MAX); a rasterised
grid was rejected because preserving the ±3mm signal would need ~88k cells,
while this keeps millimetre precision in ~500 floats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atrium_sim.blueprint import Blueprint, BrickKind
from atrium_sim.constants import C_MAX, H_MAX, L_MAX, MATCH_GATE_RAD, S_MAX

N_SLOT_FEATURES = 8
N_GLOBALS = 10
OBS_DIM = C_MAX * S_MAX * N_SLOT_FEATURES + N_GLOBALS  # 538

_ERR_NORM_MM = 30.0  # dx/dy normaliser: full resolution around ±3mm, clips at ±30


@dataclass(frozen=True)
class GlobalState:
    """Episode-level scalars the slot tensor can't carry (all env-owned)."""

    cursor: int                 # course being placed into
    course_fill_frac: float     # matched / total targets in cursor course
    bricks_left: int
    budget: int
    cuts: int                   # half-brick placements so far
    n_strays: int
    last_disturbance_mm: float  # max centre displacement of pre-existing bricks, last settle
    next_slot_x: float          # x* of leftmost unmatched target in cursor course (0 if none)
    next_slot_is_half: float    # 1.0 if that slot wants a HALF brick (the a[1] signal)


def encode(blueprint: Blueprint, report, g: GlobalState) -> np.ndarray:
    """Flattened (538,) float32 observation in [-1, 1]."""
    slots = np.zeros((C_MAX, S_MAX, N_SLOT_FEATURES), dtype=np.float32)
    match_by_target = {m.target_id: m for m in report.matches}
    for t in blueprint.targets:
        m = match_by_target.get(t.tid)
        slots[t.course, t.slot] = (
            1.0,
            1.0 if t.kind == BrickKind.HALF else 0.0,
            t.x / L_MAX,
            t.y / H_MAX,
            1.0 if m is not None else 0.0,
            # unfilled slots keep dx=dy=theta=0; `filled` is the discriminator
            np.clip(m.dx / _ERR_NORM_MM, -1.0, 1.0) if m else 0.0,
            np.clip(m.dy / _ERR_NORM_MM, -1.0, 1.0) if m else 0.0,
            np.clip(m.dtheta / MATCH_GATE_RAD, -1.0, 1.0) if m else 0.0,
        )
    n = blueprint.n_targets
    globals_vec = np.array(
        [
            g.cursor / C_MAX,
            g.course_fill_frac,
            g.bricks_left / g.budget,
            g.cuts / n,
            g.n_strays / n,
            min(g.last_disturbance_mm / _ERR_NORM_MM, 1.0),
            blueprint.length / L_MAX,
            blueprint.n_courses / C_MAX,
            g.next_slot_x / L_MAX,
            g.next_slot_is_half,   # lets the policy choose full/half for the suggested slot
        ],
        dtype=np.float32,
    )
    obs = np.concatenate([slots.reshape(-1), globals_vec])
    return np.clip(obs, -1.0, 1.0)
