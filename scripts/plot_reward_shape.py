"""Plot the two closed-form shapes behind the reward story, no training logs needed:

  - reward_quality_curve.png: the audit's plateau + Gaussian shoulder (atrium_sim.reward.
    plateau_gauss) at the shipped sigma_mm=6/sigma_deg=2 vs the dataclass default sigma_mm=12,
    for position and angle - "full credit inside +-3mm/+-0.5 deg, smooth decay outside".
  - gradient_desert.png: why absolute placement is a gradient desert and slot-relative isn't -
    the fraction of the ACTION's own reachable range that falls inside the 55mm match gate,
    for absolute mode (shrinks as the wall grows) vs slot-relative mode (100% by construction,
    since offset_range_mm=15 < match_gate_mm=55 for every wall size).

    uv run python scripts/plot_reward_shape.py --out media/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from atrium_sim.blueprint import SIZE_LADDER
from atrium_sim.constants import MATCH_GATE_MM, MODULE_MM, JOINT_MM, OFFSET_RANGE_MM, TOL_DEG, TOL_MM
from atrium_sim.reward import plateau_gauss

# same tokens as the other plot_*.py scripts - one figure language across the repo
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
ACCENT, REFERENCE = "#2a78d6", "#eb6834"


def style_axes(ax) -> None:
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_quality_curve(out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150)

    d = np.linspace(0, 30, 400)
    for sigma, color, label in ((12.0, MUTED, "default (sigma_mm=12)"),
                                (6.0, ACCENT, "shipped (sigma_mm=6)")):
        q = [plateau_gauss(v, TOL_MM, sigma) for v in d]
        ax1.plot(d, q, color=color, lw=2, zorder=3, label=label)
    ax1.axvspan(0, TOL_MM, color=ACCENT, alpha=0.08, zorder=1)
    ax1.axvline(TOL_MM, color=GRID, lw=1, ls=(0, (4, 3)), zorder=2)
    ax1.set_title("Position quality", color=INK, fontsize=11, loc="left")
    ax1.set_xlabel("deviation from target (mm)", color=MUTED, fontsize=9)
    ax1.set_ylabel("quality s_pos", color=MUTED, fontsize=9)
    ax1.legend(loc="upper right", frameon=False, fontsize=8.5)

    # sigma_deg is 2.0 in both the dataclass default and the shipped override - one curve
    deg = np.linspace(0, 6, 400)
    q = [plateau_gauss(v, TOL_DEG, 2.0) for v in deg]
    ax2.plot(deg, q, color=ACCENT, lw=2, zorder=3, label="sigma_deg=2 (default = shipped)")
    ax2.axvspan(0, TOL_DEG, color=ACCENT, alpha=0.08, zorder=1)
    ax2.axvline(TOL_DEG, color=GRID, lw=1, ls=(0, (4, 3)), zorder=2)
    ax2.set_title("Angle quality", color=INK, fontsize=11, loc="left")
    ax2.set_xlabel("deviation from target (deg)", color=MUTED, fontsize=9)
    ax2.set_ylabel("quality s_ang", color=MUTED, fontsize=9)
    ax2.legend(loc="upper right", frameon=False, fontsize=8.5)

    for ax in (ax1, ax2):
        style_axes(ax)
        ax.set_ylim(-0.02, 1.05)
        ax.margins(x=0.01)
    fig.suptitle("The audit's plateau + Gaussian shoulder (q = s_pos * s_ang)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out / "reward_quality_curve.png"
    fig.savefig(out_path)
    print("wrote", out_path)


def plot_gradient_desert(out: Path) -> None:
    n_modules = np.array(sorted({m for m, _ in SIZE_LADDER} | set(range(3, 21))))
    lengths = MODULE_MM * n_modules - JOINT_MM

    # absolute mode: a[0] in [-1,1] maps LINEARLY across the whole wall length - the fraction
    # of that range landing within the match gate of any one fixed target is the gate's own
    # width (2x MATCH_GATE_MM) over the wall length, clipped at 1.0 for tiny walls.
    frac_absolute = np.minimum(1.0, 2 * MATCH_GATE_MM / lengths)
    # slot-relative mode: a[0] maps across +-OFFSET_RANGE_MM CENTERED ON THE TARGET ITSELF, so
    # the entire action range sits inside the gate whenever OFFSET_RANGE_MM <= MATCH_GATE_MM -
    # a structural guarantee, not a probability, and independent of wall length.
    frac_slot = np.full_like(lengths, min(1.0, MATCH_GATE_MM / OFFSET_RANGE_MM), dtype=float)
    frac_slot = np.clip(frac_slot, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    ax.plot(n_modules, frac_slot * 100, color=ACCENT, lw=2, zorder=3, marker="o",
            markersize=4, label="slot-relative (+-15mm around the true slot)")
    ax.plot(n_modules, frac_absolute * 100, color=REFERENCE, lw=2, zorder=3, marker="o",
            markersize=4, label="absolute (spans the whole wall)")
    ax.set_ylim(-4, 104)
    ax.set_title("Fraction of the action's own range inside the 55mm match gate",
                 color=INK, fontsize=12, loc="left")
    ax.set_xlabel("wall width (modules)", color=MUTED, fontsize=9)
    ax.set_ylabel("% of the action range that lands in-gate", color=MUTED, fontsize=9)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    style_axes(ax)
    ax.margins(x=0.02)
    fig.tight_layout()
    out_path = out / "gradient_desert.png"
    fig.savefig(out_path)
    print("wrote", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("media"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_quality_curve(args.out)
    plot_gradient_desert(args.out)


if __name__ == "__main__":
    main()
