"""FastAPI wrapper around webviz/server.py - the OPTIONAL live backend behind the static
frontend. Deployed as a Hugging Face Space (Gradio SDK, on ZeroGPU hardware - see
deploy/space/app.py for how this app is mounted inside a minimal gr.Blocks, and
deploy/space/README.md for the one-time setup); .github/workflows/space.yml pushes updates
on every push.

Not required for the site to work: everything linked from the home page is precomputed by
scripts/export_traces.py and served as flat files. This exists only for what a static export
can't cover - an arbitrary seed/policy/spec combo, and the /build grid editor, whose plans
are drawn on the fly and can't be baked in advance.

    uv sync --extra serve --extra infer
    uv run uvicorn webviz.api:app --host 0.0.0.0 --port 7860

CORS is locked to the deployed frontend's origin (ALLOWED_ORIGINS below) at the application
level - this is real and effective when running standalone (`uvicorn webviz.api:app`, e.g.
a future Docker-SDK deployment). It has no browser-enforced effect on the actual Gradio-SDK
Space deployment: confirmed HF's own edge unconditionally adds
`Access-Control-Allow-Origin: <the request's Origin>` to every response from a public
`*.hf.space` domain, for any path and regardless of what the application decided - Spaces
are meant to be callable/embeddable from anywhere, by platform design. Accepted as a
low-severity gap (see deploy/space/app.py's `_merge_api_routes` docstring): there's no
sensitive data here, and the protections that actually matter for abuse - the policy/spec
whitelist below, the plan-size clamp, the concurrency semaphore - are all still fully
effective regardless of what the edge does with CORS headers.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

# webviz/server.py's RUNS_DIR/ROBOT_RUNS_DIR/PLANS_DIR are resolved relative to the CWD at
# call time - chdir to the repo root before anything below imports it, regardless of how or
# from where uvicorn was launched (a Space's container WORKDIR, systemd, a plain shell, ...).
os.chdir(Path(__file__).resolve().parent.parent)

from webviz.episode import SCENARIOS, SPECS  # noqa: E402
from webviz.server import (  # noqa: E402
    list_checkpoints,
    list_house_plans,
    list_robot_checkpoints,
    run_episode,
    run_robot_episode,
)

ALLOWED_ORIGINS = [
    "https://brunorosilva.github.io",
    "http://localhost:3000",  # `npm run dev` against a local API, NEXT_PUBLIC_API_BASE set
    *([os.environ["EXTRA_ALLOWED_ORIGIN"]] if os.environ.get("EXTRA_ALLOWED_ORIGIN") else []),
]

app = FastAPI(title="atrium-sim webviz API")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)

# One episode at a time. Each is real (blocking) CPU work - physics settle substeps plus a
# forward pass per step, seconds to ~a minute for a full facade - and the handlers below are
# plain `def`s (not `async def`s) specifically so FastAPI/Starlette runs them in its
# threadpool instead of stalling the event loop; this semaphore then caps how many of that
# threadpool's workers can be doing episode work at once. CPU Basic (the Space's hardware
# tier) has 2 vCPU to share - a handful of concurrent visitors would otherwise pin both.
_episode_slot = threading.Semaphore(1)


@app.on_event("startup")
def _warm_caches() -> None:
    """list_robot_checkpoints() torch.loads every candidate checkpoint the first time it's
    called and memoizes the result by (path, mtime) - see webviz/server.py's own docstring
    on why that's cached rather than skipped. Pay that once here, at boot, instead of making
    whichever visitor's request lands first eat it. list_checkpoints() is a plain glob and
    costs nothing; it's called alongside only because /policies needs both."""
    list_robot_checkpoints()
    list_checkpoints()


class EpisodeRequest(BaseModel):
    env: str = "robot"
    policy: str = "oracle"
    seed: int = 0
    spec: str = "random"
    scenario: str = "empty"
    plan: Optional[dict] = None  # a /build grid-editor plan; overrides spec/scenario when set


def _valid_robot_policies() -> set[str]:
    return {"oracle", "random"} | {f"ckpt:{c}" for c in list_robot_checkpoints()}


def _valid_bricklayer_policies() -> set[str]:
    return {"oracle", "greedy", "random"} | {f"ckpt:{c}" for c in list_checkpoints()}


def _valid_specs(env: str) -> set[str]:
    return (set(SPECS) | set(list_house_plans())) if env == "robot" else set(SPECS)


# The /build grid editor caps its own inputs (2-40 modules and courses, ring depth <= 6 -
# see frontend/components/GridEditor.tsx), but that's client-side on an endpoint that's open
# on the public internet, and FacadePlan.validate() only checks in-grid/no-overlap, never
# SIZE. Without a server-side ceiling, {"plan": {"grid_cols": 5000, "grid_rows": 5000}}
# tiles millions of modules, gets a proportionally enormous step budget (robot_env.py's
# self.budget), accumulates a pose snapshot per physics tick in RAM, and holds
# _episode_slot for the whole ride - one request wedges the Space for every visitor until
# the container OOMs. Same ceilings as the editor, enforced here where they can't be
# bypassed.
MAX_GRID = 40
MAX_OPENINGS = 24
MAX_RING_COURSES = 6
MAX_PANELS = 2 * MAX_GRID * MAX_GRID  # validate()'s overlap check is O(n^2) over panels


def _plan_rejection(plan: dict) -> str | None:
    """None if the plan is within the editor's own limits, else why it isn't."""
    try:
        cols, rows = int(plan["grid_cols"]), int(plan["grid_rows"])
    except (KeyError, TypeError, ValueError):
        return "plan needs integer grid_cols/grid_rows"
    if not (1 <= cols <= MAX_GRID and 1 <= rows <= MAX_GRID):
        return f"grid must be 1x1..{MAX_GRID}x{MAX_GRID} (got {cols}x{rows})"
    openings = plan.get("openings") or []
    if not isinstance(openings, list) or len(openings) > MAX_OPENINGS:
        return f"at most {MAX_OPENINGS} openings"
    for o in openings:
        if not isinstance(o, dict):
            return "each opening must be an object"
        try:
            ring = int(o.get("arch_ring_courses") or 1)
        except (TypeError, ValueError):
            return "arch_ring_courses must be an integer"
        if ring > MAX_RING_COURSES:
            return f"arch_ring_courses must be <= {MAX_RING_COURSES}"
    panels = plan.get("panels") or []
    if not isinstance(panels, list) or len(panels) > MAX_PANELS:
        return f"at most {MAX_PANELS} panels"
    return None


@app.get("/")
def root():
    return {"service": "atrium-sim webviz API", "docs": "/docs"}


@app.get("/policies")
def policies(env: str = "robot"):
    if env == "robot":
        return {"policies": sorted(_valid_robot_policies()), "specs": SPECS + list_house_plans(),
                "scenarios": ["empty", "prefill"]}
    if env == "bricklayer":
        return {"policies": sorted(_valid_bricklayer_policies()), "specs": SPECS, "scenarios": SCENARIOS}
    raise HTTPException(400, f"unknown env: {env!r}")


@app.post("/episode")
def episode(req: EpisodeRequest):
    # Whitelist-validate everything up front rather than letting an unrecognized policy/spec
    # fall through to build_policy/build_robot_policy, which resolve a `ckpt:<name>` straight
    # into a filesystem path - see webviz/server.py's own traversal guard on the house-plan
    # side; this is the same protection for the (otherwise unguarded) checkpoint side, on an
    # endpoint that's now open on the public internet instead of a Tailscale-only LAN.
    if req.env not in ("robot", "bricklayer"):
        return {"error": f"unknown env: {req.env!r}"}
    if req.plan is not None:
        if req.env != "robot":
            return {"error": "custom plans are robot-env only"}
        if req.policy not in _valid_robot_policies():
            return {"error": f"unknown policy: {req.policy!r}"}
        if (why := _plan_rejection(req.plan)) is not None:
            return {"error": why}
    else:
        valid_policies = _valid_robot_policies() if req.env == "robot" else _valid_bricklayer_policies()
        if req.policy not in valid_policies:
            return {"error": f"unknown policy: {req.policy!r}"}
        if req.spec not in _valid_specs(req.env):
            return {"error": f"unknown spec: {req.spec!r}"}

    with _episode_slot:
        try:
            if req.env == "robot":
                plan_json = json.dumps(req.plan) if req.plan is not None else None
                return run_robot_episode(req.policy, req.seed, req.spec, req.scenario, plan_json=plan_json)
            return run_episode(req.policy, req.seed, req.spec, req.scenario)
        except Exception as e:  # surface the error to the browser instead of an opaque 500
            return {"error": f"{type(e).__name__}: {e}"}
