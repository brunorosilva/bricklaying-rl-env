"""One-shot episode generator for the Next.js frontend.

Prints a single replay (or the policy/spec list) as JSON to stdout, then exits.
The Next API routes spawn this per request, so every episode runs the CURRENT
env code - there is no long-lived Python process to go stale.

    python -m webviz.episode --list
    python -m webviz.episode --env robot --list
    python -m webviz.episode --policy ckpt:ppo6_... --seed 3 --spec 4x4
    python -m webviz.episode --env robot --policy oracle --spec 8x5
    echo '<FacadePlan JSON>' | python -m webviz.episode --env robot --policy oracle --plan-stdin
"""

from __future__ import annotations

import argparse
import json
import sys

from webviz.server import (
    list_checkpoints,
    list_house_plans,
    list_robot_checkpoints,
    run_episode,
    run_robot_episode,
)

SPECS = ["random", "4x2", "4x4", "5x4", "6x5", "7x3", "8x5", "10x6",
         # colonial-facade panels (watch robot15 attempt real facade pieces):
         "17x8", "7x10", "6x10", "4x13", "2x40", "1x40"]
SCENARIOS = ["empty", "prefill_base", "almost", "top_gaps"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="bricklayer", choices=["bricklayer", "robot"])
    p.add_argument("--list", action="store_true", help="print policies + specs, then exit")
    p.add_argument("--policy", default="oracle")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spec", default="random")
    p.add_argument("--scenario", default="empty")
    p.add_argument("--plan-stdin", action="store_true",
                   help="read a FacadePlan JSON from stdin (a custom grid-editor plan) "
                        "instead of resolving --spec; robot env only")
    a = p.parse_args()

    if a.env == "robot":
        plan_json = sys.stdin.read() if a.plan_stdin else None
        if a.list:
            policies = ["oracle", "random"] + [f"ckpt:{c}" for c in list_robot_checkpoints()]
            print(json.dumps({"policies": policies, "specs": SPECS + list_house_plans(),
                              "scenarios": ["empty", "prefill"]}))
        else:
            print(json.dumps(
                run_robot_episode(a.policy, a.seed, a.spec, a.scenario, plan_json=plan_json)
            ))
        return

    if a.list:
        policies = ["oracle", "greedy", "random"] + [f"ckpt:{c}" for c in list_checkpoints()]
        print(json.dumps({"policies": policies, "specs": SPECS, "scenarios": SCENARIOS}))
    else:
        print(json.dumps(run_episode(a.policy, a.seed, a.spec, a.scenario)))


if __name__ == "__main__":
    main()
