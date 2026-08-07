"""Plot the ladder (+ retrained equivalents) against the real uk_terrace facade -
a harder, structurally different project than the flat-wall eval suite "The ladder"
section's table uses. Reads scripts/eval_house_ladder.py's JSON output.

Colors match each policy's series color in the existing episodic_return.png /
frac_in_tol.png ladder charts (color follows the entity across both figures).

    uv run python scripts/eval_house_ladder.py --episodes 30 --out media/house_eval.json
    uv run python scripts/plot_house_eval.py media/house_eval.json --out media/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# same hue assigned to each policy in plot_curves.py's --include robot5 robot8 robot11
# robot16 robot18 ordering - robot8_v2/11_v2/16_v2 collapse robot5/7/8 -> robot8's slot.
COLORS = {
    "robot8_v2": "#eb6834",
    "robot11_v2": "#1baf7a",
    "robot16_v2": "#a855f7",
    "robot18": "#d6336c",
}
LABELS = {
    "robot8_v2": "robot8_v2\n(small suite)",
    "robot11_v2": "robot11_v2\n(drop-control)",
    "robot16_v2": "robot16_v2\n(size curriculum)",
    "robot18": "robot18\n(shipped)",
}
METRICS = [
    ("frac_filled", "fill"),
    ("frac_in_tol", "within ±3mm"),
    ("ring_closure", "ring closure"),
    ("arch_strike_survival", "strike survival"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=Path("media"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.results.read_text())
    policies = [p for p in COLORS if p in data]

    n_groups, n_bars = len(METRICS), len(policies)
    width = 0.8 / n_bars
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=150)

    for i, policy in enumerate(policies):
        xs = [g + (i - (n_bars - 1) / 2) * width for g in range(n_groups)]
        ys = [data[policy].get(key, 0.0) for key, _ in METRICS]
        color = COLORS[policy]
        bars = ax.bar(xs, ys, width=width * 0.92, color=color, zorder=3, label=policy)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y:.0%}", (b.get_x() + b.get_width() / 2, y),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", color=INK, fontsize=7.5)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([label for _, label in METRICS], color=MUTED, fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=8.5)
    ax.set_title("The ladder vs. a real project: the uk_terrace facade (30 held-out episodes)",
                 color=INK, fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)

    legend = ax.legend([LABELS[p] for p in policies], loc="upper left", frameon=False,
                       fontsize=8, ncol=1, bbox_to_anchor=(1.0, 1.0))
    for text, policy in zip(legend.get_texts(), policies):
        text.set_color(COLORS[policy])
    fig.tight_layout()

    out = args.out / "house_eval_ladder.png"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
