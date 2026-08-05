---
title: Atrium Sim Webviz
emoji: 🧱
colorFrom: orange
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Live backend for the bricklaying-RL frontend - runs one episode per request
---

# atrium-sim webviz API

The optional live backend for [the atrium-sim frontend](https://brunorosilva.github.io/bricklaying-rl-env/)
(`webviz/api.py`, wrapping `webviz/server.py`). **Not required for that site to work** -
every case linked from its home page is a precomputed replay baked by
`scripts/export_traces.py` and served as flat files, so the frontend stays fully usable if
this Space is asleep, unreachable, or never configured at all.

This exists for the two things a static export can't cover:
- an arbitrary seed for any policy/wall-size combination,
- the `/build` grid editor, whose plans are drawn on the fly and can't be baked in advance.

## Endpoints

- `GET /policies?env=robot|bricklayer` - the same policy/spec/scenario lists
  `webviz/episode.py --list` prints, live off whatever checkpoints are present under `runs/`.
- `POST /episode` - `{"env", "policy", "seed", "spec", "scenario", "plan"?}` -> a replay JSON
  (see `frontend/lib/replay/types.ts`), or `{"error": "..."}` for an invalid combination.

CORS is restricted to `https://brunorosilva.github.io` (see `webviz/api.py`'s
`ALLOWED_ORIGINS`) - it isn't meant to be called from arbitrary sites, since each request
runs a real (if CPU-cheap) episode.

## How this gets here

This whole directory tree is **not** the main GitHub repo - it's a subtree staged and
pushed by `.github/workflows/space.yml` on every push to `main` (only what
`deploy/space/Dockerfile` needs: `atrium_sim/`, `webviz/`, `train/`, `baselines/`,
`plans/`, `runs/**/ckpt.pt`, `pyproject.toml`, `uv.lock`, plus this file as the Space's
`README.md` and `deploy/space/Dockerfile` as its `Dockerfile`). See that workflow for the
one-time setup (`HF_TOKEN` + `HF_SPACE_REPO` repo secrets/variables).

Creating a Docker Space requires a paid plan - PRO for a personal account, Team or
Enterprise for an organization; Static Spaces are the only free-to-create kind. The
hardware itself is cheap: CPU Basic (2 vCPU / 16GB RAM / 50GB non-persistent disk) carries
no hourly charge once you have the plan. On CPU Basic a Space sleeps after 48h with no
traffic and that timer isn't configurable at this tier - the first request afterwards
wakes the container, which takes ~30-60s. The frontend's live-fallback path
(`frontend/lib/traces.ts`) accounts for both halves of that: a waking Space answers with
HF's own HTML page rather than JSON, which is checked for explicitly, and the request
carries a timeout so a Space that isn't answering at all surfaces the same "try again
shortly" message instead of hanging the UI.

`EXTRA_ALLOWED_ORIGIN` (read at `webviz/api.py`'s `ALLOWED_ORIGINS`) can be set as a
runtime **Variable** in the Space's own Settings (not a GitHub variable) if you ever need
to allow a second origin - e.g. testing against a custom domain.
