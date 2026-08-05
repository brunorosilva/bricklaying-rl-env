"""FastAPI wrapper around webviz/server.py - the OPTIONAL live backend behind the static
frontend. Deployed as a Hugging Face Space (Docker SDK); see deploy/space/README.md for the
one-time setup and .github/workflows/space.yml for how it auto-updates on push.

Not required for the site to work: everything linked from the home page is precomputed by
scripts/export_traces.py and served as flat files. This exists only for what a static export
can't cover - an arbitrary seed/policy/spec combo, and the /build grid editor, whose plans
are drawn on the fly and can't be baked in advance.

    uv sync --extra serve --extra train
    uv run uvicorn webviz.api:app --host 0.0.0.0 --port 7860

CORS is locked to the deployed frontend's origin (ALLOWED_ORIGINS below), not left open to
arbitrary sites - each request runs a real (if CPU-cheap, seconds-to-a-minute) episode.
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
# threadpool's workers can be doing episode work at once. Free-tier Space CPUs have very
# little to share - a handful of concurrent visitors would otherwise pin every core.
_episode_slot = threading.Semaphore(1)


@app.on_event("startup")
def _warm_caches() -> None:
    """list_robot_checkpoints()/list_checkpoints() torch.load every candidate checkpoint the
    FIRST time either is called (see webviz/server.py's own docstring on why that's cached
    by mtime rather than skipped) - pay that cost once here, at boot, instead of making
    whichever visitor's request happens to land first eat it."""
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
