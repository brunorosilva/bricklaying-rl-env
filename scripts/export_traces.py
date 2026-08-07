"""Bake a matrix of replays to static, gzipped JSON for the GitHub Pages deploy.

The frontend (`frontend/`) is built with `output: "export"` - no Node server, so no
`/api/episode` to run a policy on demand. But every replay is fully deterministic given
(env, policy, spec, scenario, seed) - see atrium_sim/physics.py's own docstring and
tests/test_physics_determinism.py - so the whole matrix can be baked once, here, and
served as flat files. `frontend/lib/traces.ts` is the reader; `webviz/episode.py` /
`webviz/server.py` are the same functions this reuses to run episodes live in dev.

Each trace is gzipped (~30x smaller than raw JSON on these payloads - a 4MB house replay
is ~120KB gzipped) and written as frontend/public/traces/<slug>.json.gz, alongside one
frontend/public/traces/index.json manifest the frontend loads first to know what exists.

    uv run python scripts/export_traces.py                  # full matrix
    uv run python scripts/export_traces.py --robot-specs 6x5 --policies oracle,random  # smoke test
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)  # webviz.server's RUNS_DIR/PLANS_DIR are cwd-relative; run from anywhere

sys.path.insert(0, str(REPO_ROOT))
from webviz.server import (  # noqa: E402
    list_house_plans,
    list_robot_checkpoints,
    run_episode,
    run_robot_episode,
)

# webviz.server.list_checkpoints() (base-task ckpt: policies) isn't used here: the one
# checkpoint under runs/ (ppo6) predates the current Agent architecture (renamed layers -
# actor_mean/critic vs today's actor_backbone/actor_head/critic_backbone/critic_head) and
# fails to load, unlike list_robot_checkpoints() which already filters incompatible robot
# checkpoints by obs_dim. Nothing to bake for bricklayer beyond the untrained baselines
# until a checkpoint is retrained on today's architecture.

OUT_DIR = REPO_ROOT / "frontend" / "public" / "traces"

ROBOT_SPECS = ["4x2", "6x5", "8x5", "10x6", "17x8", "house:uk_terrace"]
BRICKLAYER_SPECS = ["4x2", "4x4", "6x5", "8x5"]
BRICKLAYER_POLICIES = ["oracle", "greedy", "random"]

SPEC_LABELS = {
    "4x2": ("4×2 wall", "wall"), "4x4": ("4×4 wall", "wall"), "6x5": ("6×5 wall", "wall"),
    "8x5": ("8×5 wall - beyond reach", "wall"), "10x6": ("10×6 wall", "wall"),
    "17x8": ("17×8 facade panel", "wall"), "random": ("Surprise me", "wall"),
    "house:uk_terrace": ("UK terrace - 3 structural arches", "house"),
    "house:uk_terrace_classic": ("UK terrace, classic - jack-free, 100% fill", "house"),
    "house:colonial": ("Colonial facade - image → plan", "house"),
}


def _classify_robot_policy(name: str) -> tuple[str, str, int]:
    """(group_id, label, sort_rank) - grouping is name-pattern based since checkpoint dirs
    are timestamped and not otherwise tagged; see README for what each lineage/run means."""
    if name in ("oracle", "random"):
        return "baselines", name, 90
    run = name[len("ckpt:"):]
    if run.startswith("robot18a"):
        return "ablation", "robot18a (arch_prob_max=0)", 10
    if run.startswith("robot18"):
        return "featured", "robot18", 0
    for lineage in ("robot8_v2", "robot11_v2", "robot16_v2"):
        if run.startswith(lineage):
            return "lineage", lineage, 20
    if run.startswith("sweep_"):
        arch = run[len("sweep_"):]
        arch = arch.rsplit("_s1_", 1)[0] if "_s1_" in arch else arch
        return "bakeoff", arch, 30
    return "other", run, 80


def _slug(s: str) -> str:
    s = s.lower().replace("ckpt:", "ckpt-").replace("house:", "house-")
    return re.sub(r"[^a-z0-9._-]+", "-", s).strip("-")


def _pkg_version(mod: str) -> str:
    m = __import__(mod)
    for attr in ("__version__", "version"):
        v = getattr(m, attr, None)
        if v:
            return str(v)
    return "?"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()[:12]
    except Exception:
        return "unknown"


class Baker:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.traces: dict[str, dict] = {}
        self.seen_slugs: set[str] = set()
        self.skipped: list[tuple[str, str]] = []

    def bake(self, env: str, policy: str, spec: str, scenario: str, seed: int, *, label_group=None):
        key = f"{env}|{policy}|{spec}|{scenario}|{seed}"
        if key in self.traces:
            return
        slug = _slug(f"{env}__{policy}__{spec}__{scenario}__s{seed}")
        assert slug not in self.seen_slugs, f"slug collision: {slug!r} (from key {key!r})"
        self.seen_slugs.add(slug)

        t0 = time.time()
        try:
            data = (run_robot_episode(policy, seed, spec, scenario) if env == "robot"
                    else run_episode(policy, seed, spec, scenario))
        except Exception as e:
            print(f"  SKIP {key}: {type(e).__name__}: {e}")
            self.skipped.append((key, f"{type(e).__name__}: {e}"))
            return
        dt = time.time() - t0

        raw = json.dumps(data, separators=(",", ":")).encode()
        gz = gzip.compress(raw, compresslevel=9)
        (self.out_dir / f"{slug}.json.gz").write_bytes(gz)

        m = data.get("metrics", {})
        self.traces[key] = {
            "file": f"{slug}.json.gz",
            "env": env, "policy": policy, "spec": spec, "scenario": scenario, "seed": seed,
            "gz_bytes": len(gz), "raw_bytes": len(raw),
            "steps": len(data["steps"]), "ticks": sum(len(s["ticks"]) for s in data["steps"]),
            "truncated": bool(data.get("truncated", False)),
            "metrics": m,
        }
        print(f"  {key}  {len(gz)/1024:.0f}KB gz  ({dt:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--robot-specs", default=",".join(ROBOT_SPECS))
    ap.add_argument("--bricklayer-specs", default=",".join(BRICKLAYER_SPECS))
    ap.add_argument("--policies", default=None,
                     help="comma-separated robot policy override (default: every compatible "
                          "checkpoint under runs/robot + runs/*sweep*, plus oracle/random)")
    ap.add_argument("--no-extras", action="store_true",
                     help="skip robot18's house:colonial / random-seed / prefill extras")
    args = ap.parse_args()

    robot_specs = [s for s in args.robot_specs.split(",") if s]
    bricklayer_specs = [s for s in args.bricklayer_specs.split(",") if s]

    if args.policies:
        robot_policies = args.policies.split(",")
    else:
        robot_policies = ["oracle", "random"] + [f"ckpt:{c}" for c in list_robot_checkpoints()]
    print(f"robot policies ({len(robot_policies)}): {robot_policies}")
    print(f"robot specs ({len(robot_specs)}): {robot_specs}")

    baker = Baker(args.out)

    print("\n== robot matrix ==")
    for policy in robot_policies:
        for spec in robot_specs:
            baker.bake("robot", policy, spec, "empty", 0)

    featured = next((p for p in robot_policies if _classify_robot_policy(p)[0] == "featured"), None)
    if featured and not args.no_extras:
        print(f"\n== {featured} extras ==")
        for name in list_house_plans():
            if name == "house:uk_terrace":
                continue  # already in the base matrix
            for seed in (0, 1, 2):
                baker.bake("robot", featured, name, "empty", seed)
        for seed in range(5):
            baker.bake("robot", featured, "random", "empty", seed)
        for spec in ("6x5", "10x6"):
            baker.bake("robot", featured, spec, "prefill", 0)

    print("\n== bricklayer matrix ==")
    for policy in BRICKLAYER_POLICIES:
        for spec in bricklayer_specs:
            baker.bake("bricklayer", policy, spec, "empty", 0)

    policy_groups: dict[str, dict] = {}
    for p in robot_policies:
        gid, label, rank = _classify_robot_policy(p)
        g = policy_groups.setdefault(gid, {"id": gid, "rank": rank, "policies": []})
        entry_label = label if gid in ("baselines",) else f"{label} ({p.split(':', 1)[-1]})" if gid == "bakeoff" else label
        g["policies"].append({"id": p, "label": entry_label})
    ordered_groups = sorted(policy_groups.values(), key=lambda g: g["rank"])
    for g in ordered_groups:
        g.pop("rank")

    manifest = {
        "schema": 1,
        "git_sha": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {
            "python": platform.python_version(),
            "pymunk": _pkg_version("pymunk"),
            "numpy": _pkg_version("numpy"),
        },
        "featured_policy": {"robot": featured or "oracle", "bricklayer": "oracle"},
        "specs": {s: {"label": lbl, "kind": kind} for s, (lbl, kind) in SPEC_LABELS.items()},
        "robot_specs": robot_specs,
        "bricklayer_specs": bricklayer_specs,
        "policy_groups": ordered_groups,
        "bricklayer_policies": BRICKLAYER_POLICIES,
        "traces": baker.traces,
        "skipped": [{"key": k, "error": e} for k, e in baker.skipped],
    }
    (args.out / "index.json").write_text(json.dumps(manifest, indent=1))

    total_gz = sum(t["gz_bytes"] for t in baker.traces.values())
    print(f"\nwrote {len(baker.traces)} traces ({total_gz/1024/1024:.1f}MB gz total), "
          f"{len(baker.skipped)} skipped -> {args.out}")
    if baker.skipped:
        print("skipped:")
        for k, e in baker.skipped:
            print(f"  {k}: {e}")


if __name__ == "__main__":
    main()
