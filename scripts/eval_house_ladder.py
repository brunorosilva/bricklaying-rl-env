"""Evaluate the robot ladder on the real uk_terrace facade - a harder, more complex
project than the flat-wall eval suite the ladder table (README, "The ladder" section)
was measured on.

robot5/robot7/robot8's distinguishing bugs (diagonal build order, learned brick kind) are
now permanent, unconditional fixes in the env code, not flags - so under today's code all
three collapse into one recipe. That leaves four checkpoints with the SAME 28-dim, mask-
capable observation the current env produces, each retrained (or, for robot18, reused
as-is) at its original recipe/step-count:

    robot8_v2   (2M steps,  suite=robot,     no random_start, no drop_control)
    robot11_v2  (10M steps, suite=robot_big, random_start, drop_control)
    robot16_v2  (4M steps,  curriculum cap=3, random_start)
    robot18     (6M steps,  curriculum cap=6, random_start, arch/scenario mix - unchanged)

Each policy is evaluated with ITS OWN training-time random_start setting (matching how
ppo_robot.py's own eval_house logging works) - scoring a policy on a start distribution it
never trained on isn't a fair generalization test.

Metrics are logged to a fresh TensorBoard run per checkpoint (same sourcing convention as
every other figure in this README) and written to a JSON summary.

    uv run python scripts/eval_house_ladder.py --episodes 30 --out media/house_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import atrium_sim  # noqa: F401
from train.agent import load_hybrid_agent
from train.ppo_robot import evaluate_robot_house

# (label, random_start used during that policy's own training)
LADDER = [
    ("robot8_v2", False),
    ("robot11_v2", True),
    ("robot16_v2", True),
    ("robot18", True),
]


def latest_ckpt(run_root: Path, label: str) -> Path:
    matches = sorted(run_root.glob(f"{label}_mlp_s*/ckpt.pt"))
    if not matches:
        raise SystemExit(f"no checkpoint found for {label!r} under {run_root}")
    return matches[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=Path("runs/robot"))
    ap.add_argument("--plan", type=Path, default=Path("plans/uk_terrace.json"))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--sigma-mm", type=float, default=6.0)
    ap.add_argument("--sigma-deg", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path("media/house_eval.json"))
    ap.add_argument("--tb-out", type=Path, default=Path("runs/robot_house_eval"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these labels (default: every entry in LADDER)")
    args = ap.parse_args()

    from torch.utils.tensorboard import SummaryWriter

    stamp = int(time.time())
    results: dict[str, dict] = {}
    for label, random_start in LADDER:
        if args.only and label not in args.only:
            continue
        ckpt = latest_ckpt(args.run_root, label)
        print(f"{label}: loading {ckpt}")
        agent = load_hybrid_agent(str(ckpt))
        metrics = evaluate_robot_house(agent, args.episodes, args.sigma_mm, args.sigma_deg,
                                       random_start, plan_path=str(args.plan))
        results[label] = metrics
        summary = ", ".join(f"{k}={v:.2%}" for k, v in metrics.items())
        print(f"{label}: {summary}")

        writer = SummaryWriter(str(args.tb_out / f"{label}_mlp_s1_{stamp}"))
        for k, v in metrics.items():
            writer.add_scalar(f"eval_house/{k}", v, 0)
        writer.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
