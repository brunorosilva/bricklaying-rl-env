"""Plot the mobile robot's held-out eval curves from a TensorBoard log.

Three figures from one run's log:
  - <name> (default robot18_eval_curves.png): the headline "before vs after" story for the
    stops-in-place diagnosis - three commensurate fraction metrics (all in [0, 1], same axis,
    not a dual-axis chart) climbing from near-zero to near-perfect over training - in-tolerance
    precision on the held-out huge-wall suite, the oracle-gated scenario library's own score,
    and structural-arch strike survival.
  - <run>_curriculum.png: the competence-gated size curriculum - the rung reached (indexed to
    [0, 1] as level/cap, not a second y-axis) alongside frontier fill, so both series share one
    axis instead of a dual-axis chart.
  - <run>_scenarios.png: a dumbbell (early eval vs final eval) per scenario-library family -
    the form that reads cleanly with 9+ categories instead of 9+ competing hues. Sorted by
    improvement, so the one family that had to be learned floats to the top.

    uv run python scripts/plot_robot_curves.py runs/robot/robot18_mlp_s1_<ts> --out media/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tbparse import SummaryReader

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
ACCENT, REFERENCE, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (
    ("eval/frac_in_tol", ACCENT, "in ±3mm (held-out huge-wall eval)"),
    ("eval_scenarios/mean", REFERENCE, "oracle-gated scenario library"),
    ("eval_arch/arch_strike_survival", THIRD, "arch strike survival"),
)


def style_axes(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_eval_curves(df, out: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    plotted = False
    for tag, color, label in SERIES:
        sub = df[df["tag"] == tag].sort_values("step")
        if not len(sub):
            print(f"warning: no data for {tag}")
            continue
        plotted = True
        ax.plot(sub["step"], sub["value"], color=color, lw=2, zorder=3, label=label)
    if not plotted:
        raise SystemExit("no matching scalar tags found for the eval-curves figure")

    # a legend, not end-of-line labels: all three series are fractions that converge toward
    # 1.0 by the end of training, so direct labels there collide into unreadable text
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=1.6)
    for text, (_, color, _) in zip(legend.get_texts(), SERIES):
        text.set_color(color)

    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Mobile robot: held-out competence over training", color=INK, fontsize=12, loc="left")
    ax.set_xlabel("environment steps", color=MUTED, fontsize=9)
    ax.set_ylabel("fraction", color=MUTED, fontsize=9)
    style_axes(ax)
    ax.margins(x=0.02)
    fig.tight_layout()
    out_path = out / name
    fig.savefig(out_path)
    print("wrote", out_path)


def plot_curriculum(df, out: Path, run_name: str) -> None:
    level = df[df["tag"] == "curriculum/level"].sort_values("step")
    frontier = df[df["tag"] == "curriculum/frontier_frac_filled"].sort_values("step")
    if not len(level) or not len(frontier):
        print("warning: no curriculum/* tags found - skipping the curriculum figure "
              "(this run wasn't trained with --curriculum)")
        return
    cap = max(1, int(level["value"].max()))

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    ax.step(level["step"], level["value"] / cap, where="post", color=ACCENT, lw=2, zorder=3,
            label=f"curriculum rung reached (0 = L0 ... 1 = L{cap})")
    ax.plot(frontier["step"], frontier["value"], color=REFERENCE, lw=2, zorder=3,
            label="frontier fill (competence at the current rung)")
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=1.6)
    for text, color in zip(legend.get_texts(), (ACCENT, REFERENCE)):
        text.set_color(color)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("The size curriculum: rung advances as the frontier is mastered",
                 color=INK, fontsize=12, loc="left")
    ax.set_xlabel("environment steps", color=MUTED, fontsize=9)
    ax.set_ylabel("fraction (indexed to a common [0, 1] base)", color=MUTED, fontsize=9)
    style_axes(ax)
    ax.margins(x=0.02)
    fig.tight_layout()
    out_path = out / f"{run_name}_curriculum.png"
    fig.savefig(out_path)
    print("wrote", out_path)


def plot_scenarios(df, out: Path, run_name: str) -> None:
    sub = df[df["tag"].str.startswith("eval_scenarios/") & (df["tag"] != "eval_scenarios/mean")]
    if not len(sub):
        print("warning: no eval_scenarios/* tags found - skipping the scenarios figure "
              "(this run wasn't trained with --scenario-mix)")
        return
    rows = []
    for tag, g in sub.groupby("tag"):
        g = g.sort_values("step")
        rows.append((tag.split("/", 1)[1], float(g["value"].iloc[0]), float(g["value"].iloc[-1])))
    rows.sort(key=lambda r: r[2] - r[1])  # largest improvement last -> plotted at the top

    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 1.6), dpi=150)
    ys = range(len(rows))
    for y, (_, early, final) in zip(ys, rows):
        ax.plot([early, final], [y, y], color=GRID, lw=1.5, zorder=2)
    ax.scatter([r[1] for r in rows], list(ys), color=MUTED, s=28, zorder=3, label="early in training")
    ax.scatter([r[2] for r in rows], list(ys), color=ACCENT, s=28, zorder=3, label="end of training")
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.set_xlim(-0.02, 1.05)
    ax.set_title("Scenario library: per-skill competence, early vs end of training",
                 color=INK, fontsize=12, loc="left")
    ax.set_xlabel("frac_filled", color=MUTED, fontsize=9)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=8.5, handlelength=1.2)
    for text, color in zip(legend.get_texts(), (MUTED, ACCENT)):
        text.set_color(color)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out_path = out / f"{run_name}_scenarios.png"
    fig.savefig(out_path)
    print("wrote", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("media"))
    ap.add_argument("--name", default="robot18_eval_curves.png")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = SummaryReader(str(args.run_dir)).scalars
    run_name = args.name.rsplit("_eval_curves", 1)[0].rsplit(".png", 1)[0]
    plot_eval_curves(df, args.out, args.name)
    plot_curriculum(df, args.out, run_name)
    plot_scenarios(df, args.out, run_name)


if __name__ == "__main__":
    main()
