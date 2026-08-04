"""Regenerate README training figures from TensorBoard logs.

    uv run python scripts/plot_curves.py runs/ --out media/
    uv run python scripts/plot_curves.py runs/ --baselines media/baselines.json

    # the robot5->robot18 ladder, a curated subset of runs/robot (else EVERY run
    # dir there - side quests, arch variants, architecture sweeps - would plot):
    uv run python scripts/plot_curves.py runs/robot --baselines media/baselines.json \\
        --include robot5 robot8 robot11 robot16 robot18

Runs are grouped by experiment name (``<exp>_s<seed>_<time>``): each group is
drawn as a per-seed mean with a ±1 std band. ``--include`` keeps only groups whose
experiment name is exactly one of the given names (omit to include every group
found - fine for a small `runs/` dir, unwieldy for `runs/robot`, which accumulates
one directory per experiment ever run). Baseline anchors (oracle / greedy /
random) come from a JSON file produced by ``scripts/eval_baselines.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tbparse import SummaryReader

# palette: fixed hue order, validated (dataviz skill's validate_palette.js, light+dark) for
# up to 5 categorical series; muted ink for reference anchors. A 6th group is dropped with a
# warning rather than silently cycling colors - see main().
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#a855f7", "#d6336c"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

PANELS = (
    ("charts/episodic_return", "Episodic return", "return (= audit score)"),
    ("env/frac_in_tol", "Bricks within BIM ±3mm", "fraction of blueprint in tolerance"),
)
BASELINE_KEYS = {"charts/episodic_return": "episode_return", "env/frac_in_tol": "frac_in_tol"}


def bin_curve(steps: np.ndarray, vals: np.ndarray, n_bins: int, x_max: float):
    edges = np.linspace(0, x_max, n_bins + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            xs.append(edges[b + 1])
            ys.append(vals[mask].mean())
    return np.array(xs), np.array(ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, default=Path("media"))
    parser.add_argument("--baselines", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--include", nargs="*", default=None,
                        help="exact experiment names to keep (default: every group found); "
                             "plotted in the order given, not alphabetically")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(args.runs.iterdir()):
        if d.is_dir() and (m := re.match(r"(.+)_s\d+_\d+$", d.name)):
            groups[m.group(1)].append(d)
    if not groups:
        raise SystemExit(f"no runs matching '<exp>_s<seed>_<time>' under {args.runs}")

    if args.include:
        missing = [name for name in args.include if name not in groups]
        if missing:
            raise SystemExit(f"--include names not found under {args.runs}: {missing}")
        ordered = [(name, groups[name]) for name in args.include]
    else:
        ordered = sorted(groups.items())
    if len(ordered) > len(SERIES):
        print(f"warning: {len(ordered)} groups > {len(SERIES)} validated colors - "
              f"dropping {[e for e, _ in ordered[len(SERIES):]]} rather than cycling hues")
        ordered = ordered[:len(SERIES)]

    anchors = json.loads(args.baselines.read_text()) if args.baselines else {}

    for tag, title, ylabel in PANELS:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        x_max = 0.0
        for gi, (exp, dirs) in enumerate(ordered):
            per_seed = []
            for d in dirs:
                df = SummaryReader(str(d)).scalars
                df = df[df["tag"] == tag]
                if len(df):
                    per_seed.append((df["step"].to_numpy(), df["value"].to_numpy()))
            if not per_seed:
                continue
            x_max = max(x_max, max(s.max() for s, _ in per_seed))
            binned = [bin_curve(s, v, args.bins, x_max) for s, v in per_seed]
            grid_x = binned[0][0]
            ys = np.array([np.interp(grid_x, bx, by) for bx, by in binned])
            mean, std = ys.mean(axis=0), ys.std(axis=0)
            color = SERIES[gi % len(SERIES)]
            ax.plot(grid_x, mean, color=color, lw=2, zorder=3)
            ax.fill_between(grid_x, mean - std, mean + std, color=color, alpha=0.18, lw=0)
            n_seeds = f"{len(per_seed)} seed" + ("s" if len(per_seed) != 1 else "")
            ax.annotate(f"{exp} ({n_seeds})", (grid_x[-1], mean[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=color, fontsize=9, va="center")
        key = BASELINE_KEYS[tag]
        for name, m in sorted(anchors.items(), key=lambda kv: -kv[1].get(key, 0)):
            if key in m:
                ax.axhline(m[key], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
                ax.annotate(name, (0.0, m[key]), xytext=(4, 3), textcoords="offset points",
                            color=MUTED, fontsize=8.5)
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.set_xlabel("environment steps", color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.margins(x=0.02)
        fig.tight_layout()
        out = args.out / f"{tag.split('/')[-1]}.png"
        fig.savefig(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
