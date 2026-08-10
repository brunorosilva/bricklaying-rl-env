---
title: Bricklaying with RL Webviz
emoji: 🧱
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Physics backend for a bricklaying robot built to BIM tolerance
---

# Bricklaying with RL webviz API

The optional live backend for [the Bricklaying with RL frontend](https://brunorosilva.github.io/bricklaying-rl-env/)
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
- `/ui` is a placeholder Gradio page (see "SDK" below) - not meant to be used directly.

`webviz/api.py` restricts CORS to `https://brunorosilva.github.io` at the application
level (its `ALLOWED_ORIGINS`), but this has **no browser-enforced effect on this
deployment**: HF's own edge unconditionally adds `Access-Control-Allow-Origin: <the
request's Origin>` to every response from a public `*.hf.space` domain, for any path,
regardless of what the application decides - by platform design, since Spaces are meant to
be callable/embeddable from anywhere. Accepted as a low-severity gap - there's no sensitive
data behind these endpoints, and the protections that actually matter for abuse (the
policy/spec whitelist, the plan-size clamp, the concurrency semaphore) are all
application-level and unaffected by it. See `webviz/api.py`'s and `app.py`'s own comments.

## SDK: Gradio, not Docker

Creating a **Docker**-SDK Space now requires HF PRO ($9/mo) - a policy that changed
~July 2026. A **Gradio**-SDK Space on ZeroGPU hardware remains free to create, so that's
what this is - though getting there took more than picking the SDK:

- `app.py` must call `demo.launch()` for real, not just build a `gr.Blocks` and serve it
  some other way: the `spaces` package (ZeroGPU support) monkeypatches `gr.Blocks.launch`
  to run its own startup handshake, which is what tells HF's platform "yes, a
  `@spaces.GPU` function exists" - skip `.launch()` (e.g. by using
  `gradio.mount_gradio_app` + a hand-rolled `uvicorn.run()`, an earlier version of this
  file's approach) and the Space fails at startup with "No @spaces.GPU function detected",
  even with a properly decorated function sitting unused in the module.
- `webviz/api.py`'s actual FastAPI app is grafted onto gradio's *own* app instance
  *after* `demo.launch()` returns (see `app.py`'s `_merge_api_routes`) - the reverse of
  `mount_gradio_app`, which mounts gradio into a separately-served app instead.
- `demo.launch(..., ssr_mode=False)`: Gradio 6's default on Spaces runs a Node.js SSR proxy
  in front of the Python backend that only forwards paths Gradio's own frontend knows
  about - `/policies` and `/episode` silently got the SPA shell instead of JSON otherwise,
  with no error anywhere to point at it.

This workload never touches the GPU (it's pure CPU physics simulation + small MLP forward
passes) - ZeroGPU hardware is selected purely because it's the free allowance for Gradio
Spaces, not because anything here benefits from a GPU. A hidden, unused button wired to a
`@spaces.GPU`-decorated function exists solely to satisfy the platform's startup check.

## How this gets here

This whole directory tree is **not** the main GitHub repo - it's a subtree staged and
pushed (via `huggingface_hub`'s `upload_folder`, not `git push` - HF's git remote now
rejects plain-blob binary files, and this runner doesn't have git-xet installed) by
`.github/workflows/space.yml` on every push to `main`: `atrium_sim/`, `webviz/`, `train/`,
`baselines/`, `plans/`, `runs/**/ckpt.pt`, `deploy/space/app.py` (as `app.py`),
`deploy/space/requirements.txt` (as `requirements.txt` - deliberately loose pins, see that
file's own comment on why) and this file (as `README.md`). See that workflow for the
one-time setup (`HF_TOKEN` + `HF_SPACE_REPO` repo secrets/variables).

Like any free-tier Space, this one sleeps after a period of inactivity and the first
request afterwards has to wake it. The frontend's live-fallback path
(`frontend/lib/traces.ts`) accounts for that: a waking Space can answer with an HTML page
instead of JSON, which is checked for explicitly, and the request carries a timeout so a
Space that isn't answering at all surfaces a "try again shortly" message instead of
hanging the UI.

`EXTRA_ALLOWED_ORIGIN` (read at `webviz/api.py`'s `ALLOWED_ORIGINS`) can be set as a
runtime **Variable** in the Space's own Settings (not a GitHub variable) for the
application-level allow-list - useful for a non-Spaces deployment, but (per the CORS note
above) it doesn't change what a browser actually permits when running as a Space.
