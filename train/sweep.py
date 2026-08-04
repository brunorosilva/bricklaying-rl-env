"""Architecture bake-off: train every network in train/architectures.py on the
same task, in parallel, and rank them.

Each architecture is a separate `python -m train.ppo` subprocess with its intra-op
thread count capped (so N runs share the box without oversubscribing). Because
the PyMunk env step is single-threaded per run, ~10 runs sit happily on 32 cores.

    python -m train.sweep --action-mode absolute --steps 500000
    python -m train.sweep --archs mlp cnn attention --steps 300000 --concurrency 3

Progress is visible live via TensorBoard on runs/sweep/. On completion a ranking
is written to runs/sweep/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from train.architectures import ARCHITECTURES


# matmul-bound backbones benefit from the GPU; the MLPs are env(PyMunk)-bound so
# the GPU would only add host<->device transfer overhead — keep them on CPU.
GPU_ARCHS = {"cnn", "attention", "attention2"}


def launch(arch: str, a: argparse.Namespace) -> subprocess.Popen:
    # transformers are matmul-bound (not env-bound like the MLPs), so give them
    # more intra-op threads or they bottleneck the whole sweep's wall-clock.
    threads = 6 if "attention" in arch else a.threads
    device = "cuda" if (a.robot and a.gpu and arch in GPU_ARCHS) else "cpu"
    module = "train.ppo_robot" if a.robot else "train.ppo"
    # both trainers must yield a run dir matching summarize()'s glob "sweep_{arch}_s*".
    # ppo_robot names runs "{exp}_{arch}_s..", ppo names them "{exp}_s..", so pass a
    # bare "sweep" for robot (arch gets appended) and "sweep_{arch}" for the base task.
    exp_name = "sweep" if a.robot else f"sweep_{arch}"
    cmd = [
        sys.executable, "-m", module,
        "--arch", arch,
        "--exp-name", exp_name,
        "--seed", str(a.seed),
        "--total-timesteps", str(a.steps),
        "--num-envs", str(a.num_envs),
        "--torch-threads", str(threads),
        "--sigma-mm", str(a.sigma_mm),
        "--eval-interval", str(a.eval_interval),
        "--eval-episodes", "8",
        "--gif-every", "0",
        "--run-dir", a.run_dir,
    ]
    if not a.robot:  # ppo_robot has no absolute/slot action mode
        cmd += ["--action-mode", a.action_mode]
    else:
        cmd += ["--device", device, "--eval-suite", a.eval_suite]
        if a.random_start:  # train the robot9 bidirectional task (base starts anywhere)
            cmd += ["--random-start"]
        if a.drop_control:  # train the robot11 drop-height task (box[1] = release height)
            cmd += ["--drop-control"]
        if a.async_envs:  # AsyncVectorEnv - one subprocess per env, needed to saturate cores
            cmd += ["--async-envs"]
        if a.curriculum:  # competence-gated SIZE curriculum (blueprint.SIZE_LADDER); also
            cmd += ["--curriculum", "--curriculum-cap", str(a.curriculum_cap)]  # gates arch_prob below
        if a.arch_prob_max > 0.0:  # structural arches (needs --curriculum, see ppo_robot.Args)
            cmd += ["--arch-prob-max", str(a.arch_prob_max)]
        if a.scenario_mix > 0.0:  # oracle-gated scenario library mixed into resets
            cmd += ["--scenario-mix", str(a.scenario_mix)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    log = open(Path(a.run_dir) / f"{arch}.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def run_pool(archs: list[str], a: argparse.Namespace) -> None:
    running: dict[str, subprocess.Popen] = {}
    pending = list(archs)
    done = 0
    while pending or running:
        while pending and len(running) < a.concurrency:
            arch = pending.pop(0)
            running[arch] = launch(arch, a)
            print(f"[sweep] launched {arch} ({len(running)} running, {len(pending)} queued)", flush=True)
        for arch, proc in list(running.items()):
            if proc.poll() is not None:
                done += 1
                print(f"[sweep] finished {arch} (exit {proc.returncode}, {done}/{len(archs)})", flush=True)
                del running[arch]
        time.sleep(5)


def summarize(archs: list[str], run_dir: str, robot: bool = False) -> dict:
    from tbparse import SummaryReader

    # tags that only exist when the run trained arches/scenarios (arch_prob_max/scenario_mix
    # > 0) - see train.ppo_robot's eval block. Missing gracefully -> None, not a KeyError.
    ARCH_TAGS = ("arch_strike_survival", "ring_closure", "jack_survival",
                 "semicircular_survival", "segmental_survival")

    rows = []
    for arch in archs:
        dirs = sorted(Path(run_dir).glob(f"sweep_{arch}_s*"))
        if not dirs:
            rows.append({"arch": arch, "status": "no run dir"})
            continue
        df = SummaryReader(str(dirs[-1])).scalars
        def col(tag):
            s = df[df.tag == tag].sort_values("step")
            if not len(s):
                return None, None, None
            # last5: mean of the final 5 eval points, not just the single last/max value -
            # arch_strike_survival is noisy at 3-9 eval episodes/point (see robot18's own
            # curve: bounces 0.44-1.0 throughout training), so a single spike overstates it.
            return float(s.value.max()), float(s.value.iloc[-1]), float(s.value.tail(5).mean())
        it_max, it_last, _ = col("eval/frac_in_tol")
        ret_max, ret_last, _ = col("eval/episode_return")
        fill_max, fill_last, _ = col("eval/frac_filled")
        comp_max, _, _ = col("eval/completed")
        _, sps, _ = col("charts/SPS")
        row = {
            "arch": arch, "status": "ok",
            "eval_frac_in_tol_max": it_max, "eval_frac_in_tol_last": it_last,
            "eval_frac_filled_max": fill_max, "eval_completed_max": comp_max,
            "eval_return_max": ret_max, "eval_return_last": ret_last,
            "sps": sps, "final_step": int(df.step.max()) if len(df) else 0,
        }
        if robot:
            for group in ("eval_arch", "eval_house"):
                for tag in ARCH_TAGS:
                    mx, last, last5 = col(f"{group}/{tag}")
                    row[f"{group}_{tag}_max"] = mx
                    row[f"{group}_{tag}_last5"] = last5
        rows.append(row)
    # rank by the specific, direct question this sweep asks: does the architecture make the
    # real held-out uk_terrace jack arch (the diagnosed failure) survive its strike? Falls
    # back to in-tolerance precision (the base-task/no-arch-training headline metric) when
    # house-eval tags aren't present (arch_prob_max == 0, or the base bricklayer task).
    key = "eval_house_jack_survival_last5" if robot else "eval_frac_in_tol_max"
    scored = [r for r in rows if r.get(key) is not None]
    if not scored:
        key = "eval_frac_in_tol_max"
        scored = [r for r in rows if r.get(key) is not None]
    ranked = sorted(
        scored,
        key=lambda r: (r[key], r.get("eval_arch_jack_survival_last5") or 0.0,
                       r.get("eval_frac_in_tol_max") or 0.0, r["eval_return_max"]),
        reverse=True,
    )
    return {"ranking": ranked, "all": rows, "rank_key": key}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archs", nargs="*", default=ARCHITECTURES)
    p.add_argument("--robot", action="store_true", help="sweep the mobile-robot task (hybrid head)")
    p.add_argument("--random-start", action="store_true",
                   help="robot: base starts anywhere (robot9 bidirectional task)")
    p.add_argument("--drop-control", action="store_true",
                   help="robot: model chooses release height (robot11 drop-height task)")
    p.add_argument("--gpu", action="store_true",
                   help="robot: run matmul-bound archs (cnn/attention) on CUDA")
    p.add_argument("--async-envs", action="store_true",
                   help="robot: AsyncVectorEnv (one subprocess per env) instead of sync")
    p.add_argument("--curriculum", action="store_true",
                   help="robot: competence-gated SIZE curriculum (blueprint.SIZE_LADDER)")
    p.add_argument("--curriculum-cap", type=int, default=6)
    p.add_argument("--arch-prob-max", type=float, default=0.0,
                   help="robot: fraction of curriculum episodes that build a real structural "
                        "arch instead of a flat wall (requires --curriculum)")
    p.add_argument("--scenario-mix", type=float, default=0.35,
                   help="robot: fraction of episodes drawn from the oracle-gated scenario library")
    p.add_argument("--eval-suite", default="robot_eval")
    p.add_argument("--action-mode", default="absolute", choices=["absolute", "slot_relative"])
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--sigma-mm", type=float, default=30.0)  # wide shoulder for the absolute desert
    p.add_argument("--eval-interval", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-dir", default="runs/sweep")
    a = p.parse_args()

    Path(a.run_dir).mkdir(parents=True, exist_ok=True)
    task = "robot" if a.robot else f"action_mode={a.action_mode}"
    print(f"[sweep] {len(a.archs)} archs x {a.steps} steps, {task}, concurrency={a.concurrency}",
          flush=True)
    run_pool(a.archs, a)
    summary = summarize(a.archs, a.run_dir, robot=a.robot)
    (Path(a.run_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[sweep] RANKING (by {summary['rank_key']}):", flush=True)
    for i, r in enumerate(summary["ranking"], 1):
        val = r.get(summary["rank_key"], float("nan"))
        extra = (f"filled {r.get('eval_frac_filled_max', 0):.2f}  "
                 f"house_jack {r.get('eval_house_jack_survival_last5', float('nan')):.2f}  "
                 f"house_survive {r.get('eval_house_arch_strike_survival_last5', float('nan')):.2f}  "
                 ) if a.robot else ""
        print(f"  {i:2d}. {r['arch']:14s} {summary['rank_key']} {val:.3f}  {extra}"
              f"return {r['eval_return_max']:+.2f}  ({r['sps']:.0f} SPS)", flush=True)


if __name__ == "__main__":
    main()
