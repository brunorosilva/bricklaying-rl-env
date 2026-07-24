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
from dataclasses import dataclass, field

from atrium_sim.blueprint import Blueprint, WallSpec, generate_blueprint


@dataclass(frozen=True)
class Opening:
    """A void in the brick field: cols [col, col+n_cols), rows [row, row+n_rows)."""
    kind: str
    col: int
    row: int
    n_cols: int
    n_rows: int


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
        to the grid, then tile the remaining brickwork into panels."""
        ops = tuple(_clamp_opening(o, grid_cols, grid_rows) for o in openings)
        ops = tuple(o for o in ops if o.n_cols > 0 and o.n_rows > 0)
        panels = tile_facade(grid_cols, grid_rows, ops)
        plan = cls(image_ref, grid_cols, grid_rows, ops, panels, notes)
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

def _clamp_opening(o: Opening, cols: int, rows: int) -> Opening:
    col = max(0, min(o.col, cols))
    row = max(0, min(o.row, rows))
    return Opening(o.kind, col, row, max(0, min(o.n_cols, cols - col)),
                   max(0, min(o.n_rows, rows - row)))


def _in_grid(col: int, row: int, nc: int, nr: int, cols: int, rows: int) -> bool:
    return col >= 0 and row >= 0 and nc >= 0 and nr >= 0 and col + nc <= cols and row + nr <= rows


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
