"""Plot the robot-task architecture bake-off: does swapping the policy/value backbone
change whether the mobile robot can survive the real uk_terrace jack arch (the one
diagnosed failure - ring closes fine, every architecture; the strike is what's hard)?

Two figures from the same TensorBoard logs:
  - a ranking bar chart: mean eval_house/jack_survival across ALL of training (not the
    final checkpoint alone - ppo_robot only saves the latest state each eval, so a run
    whose peak came mid-training and regressed by the end would look like a loser on a
    final-snapshot metric when it demonstrably wasn't; the training-mean survives that).
  - a trajectory chart: rolling-mean eval_house/jack_survival over steps, baseline vs the
    top non-baseline performers, showing the failure mode this metric hides at a glance -
    the skill oscillates for every architecture, never converges, for anyone.

    uv run python scripts/plot_arch_sweep.py runs/sweep_archbakeoff runs/sweep_archbakeoff_spatial --out media/
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tbparse import SummaryReader

# same tokens as scripts/plot_robot_curves.py / plot_curves.py - one figure language
# across the repo. Validated (scripts/validate_palette.js, --mode light).
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
ACCENT = "#2a78d6"
SERIES = (ACCENT, "#eb6834", "#1baf7a")  # baseline, then up to 2 challengers, fixed order

BASELINE = "mlp"  # robot18's shipped architecture - every other arch is judged against it
TAG = "eval_house/jack_survival"


def load_series(run_dirs: list[Path], arch: str) -> tuple[np.ndarray, np.ndarray] | None:
    dirs: list[Path] = []
    for run_dir in run_dirs:
        dirs += [Path(p) for p in glob.glob(str(run_dir / f"sweep_{arch}_s*"))]
    if not dirs:
        return None
    d = sorted(dirs)[-1]
    df = SummaryReader(str(d)).scalars
    s = df[df.tag == TAG].sort_values("step")
    if not len(s):
        return None
    return s["step"].to_numpy(), s["value"].to_numpy()


def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def style_axes(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("media"))
    ap.add_argument("--rolling-window", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from train.architectures import ARCHITECTURES

    means: dict[str, float] = {}
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arch in ARCHITECTURES:
        loaded = load_series(args.run_dirs, arch)
        if loaded is None:
            print(f"warning: no {TAG} data for {arch!r} - skipping (run not finished yet?)")
            continue
        series[arch] = loaded
        means[arch] = float(loaded[1].mean())

    if not means:
        raise SystemExit(f"no {TAG} data found under {args.run_dirs}")

    # --- figure 1: ranking bar chart ---
    ranked = sorted(means.items(), key=lambda kv: kv[1])  # ascending - barh reads bottom-up
    labels = [a for a, _ in ranked]
    values = [v for _, v in ranked]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(ranked) + 1.4), dpi=150)
    bars = ax.barh(labels, values, color=ACCENT, height=0.6, zorder=3)
    for b in bars:
        b.set_clip_on(False)
    for label, v in zip(labels, values):
        ax.annotate(f"{v:.0%}", (v, label), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    if BASELINE in means:
        ax.axvline(means[BASELINE], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        # headroom above the top bar (in DATA coords, not axes-fraction) so the label
        # clears both the bars and the title, whatever the bar count
        ax.set_ylim(-0.7, len(ranked) - 1 + 0.9)
        ax.annotate(f"{BASELINE} (shipped, robot18)", (means[BASELINE], len(ranked) - 1 + 0.55),
                    xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_title("Robot task: architecture vs. the real uk_terrace jack-arch strike",
                 color=INK, fontsize=12, loc="left")
    ax.set_xlabel("mean jack-arch strike survival across training (eval_house/jack_survival)",
                  color=MUTED, fontsize=9)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out1 = args.out / "arch_sweep_ranking.png"
    fig.savefig(out1)
    print("wrote", out1)

    # --- figure 2: trajectory, baseline + top challengers (rolling mean - raw survival
    # is ~binary and noisy, see the docstring; a legend, not end-labels, since a noisy
    # signal can converge/cross near the right edge) ---
    top_challengers = [a for a, _ in sorted(
        ((a, v) for a, v in means.items() if a != BASELINE), key=lambda kv: -kv[1]
    )[:2]]
    to_plot = ([BASELINE] if BASELINE in series else []) + top_challengers
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    for i, arch in enumerate(to_plot):
        if arch not in series:
            continue
        steps, vals = series[arch]
        smoothed = rolling_mean(vals, args.rolling_window)
        # 'valid' convolution output[j] = mean(vals[j : j+window]) - align each point with
        # its window's CENTER step, not the left or right edge (a real bug earlier: the
        # length-mismatch branch was taken unconditionally and always right-aligned).
        center = (args.rolling_window - 1) // 2
        xs = steps[center : center + len(smoothed)]
        label = f"{arch} (shipped baseline)" if arch == BASELINE else arch
        ax.plot(xs, smoothed, color=SERIES[i % len(SERIES)], lw=2, zorder=3, label=label)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=1.6)
    for text, arch in zip(legend.get_texts(), to_plot):
        text.set_color(SERIES[to_plot.index(arch) % len(SERIES)])
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Jack-arch survival over training ({args.rolling_window}-eval rolling mean)",
                 color=INK, fontsize=12, loc="left")
    ax.set_xlabel("environment steps", color=MUTED, fontsize=9)
    ax.set_ylabel("uk_terrace jack-arch strike survival", color=MUTED, fontsize=9)
    style_axes(ax)
    ax.margins(x=0.02)
    fig.tight_layout()
    out2 = args.out / "arch_sweep_trajectory.png"
    fig.savefig(out2)
    print("wrote", out2)


if __name__ == "__main__":
    main()
