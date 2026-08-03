"""Plot the mobile robot's held-out eval curves from a TensorBoard log.

The headline "before vs after" story for the stops-in-place diagnosis: three commensurate
fraction metrics (all in [0, 1], same axis, not a dual-axis chart) climbing from near-zero
to near-perfect over training - in-tolerance precision on the held-out huge-wall suite,
the oracle-gated scenario library's own score, and structural-arch strike survival.

    uv run python scripts/plot_robot_curves.py runs/robot/robot18_mlp_s1_<ts> --out media/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tbparse import SummaryReader

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# validated (scripts/validate_palette.js, --mode light): CVD/normal-vision separation pass;
# the teal's contrast WARN is mitigated by direct end-of-line labels, not a legend box
SERIES = (
    ("eval/frac_in_tol", "#2a78d6", "in ±3mm (held-out huge-wall eval)"),
    ("eval_scenarios/mean", "#eb6834", "oracle-gated scenario library"),
    ("eval_arch/arch_strike_survival", "#1baf7a", "arch strike survival"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("media"))
    ap.add_argument("--name", default="robot18_eval_curves.png")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = SummaryReader(str(args.run_dir)).scalars
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
        raise SystemExit(f"no matching scalar tags found under {args.run_dir}")

    # a legend, not end-of-line labels: all three series are fractions that converge toward
    # 1.0 by the end of training, so direct labels there collide into unreadable text
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=1.6)
    for text, (_, color, _) in zip(legend.get_texts(), SERIES):
        text.set_color(color)

    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Mobile robot: held-out competence over training", color=INK, fontsize=12, loc="left")
    ax.set_xlabel("environment steps", color=MUTED, fontsize=9)
    ax.set_ylabel("fraction", color=MUTED, fontsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.margins(x=0.02)
    fig.tight_layout()
    out = args.out / args.name
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
