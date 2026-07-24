# Proposal: VLM → FacadePlan (any image → buildable brick plan)

**Status:** proposed, not started (branch `feat/vlm-facade-plan` does not yet exist).
**Owner:** hand this file to a coding agent (Claude Code, Opus) or implement directly.

---

## 1. Motivation

atrium-sim currently lays a **single solid running-bond rectangle** defined by a
hand-authored `WallSpec(n_modules, n_courses)`. The goal of this work is to make the
blueprint come from **any real image** — a photo or drawing of a brick house — via a
vision-language model, so the pipeline is:

```
image (path/URL) → VLM → validated FacadePlan (JSON) → per-panel Blueprints → env/render
```

This is the "Image/VLM observations" roadmap item, and it's the software complement to
the day-job VLM blueprint-reading work: a model reads a facade, understands wall specs,
and emits a placement plan.

**Non-goal (for v1):** perfect metric accuracy from a photo. v1 proves the *conversion
path* works generally and produces a plausible, buildable, structured plan.

---

## 2. Repo facts the implementation must respect

- **Package:** `atrium_sim/` is deliberately **torch-free and gym-free** in
  `blueprint.py`/`constants.py` so the reward audit and downstream pipelines can consume
  blueprints standalone. Keep the new IR in the same import-light tier.
- **Units (`atrium_sim/constants.py`):** waalformaat brick (210×50 mm face); `MODULE_MM =
  220`, `COURSE_MM = 60`, BIM `TOL_MM = 3`. Everything is millimetres.
- **Schema (`atrium_sim/blueprint.py`):**
  - `WallSpec(n_modules: int, n_courses: int)`
  - `generate_blueprint(spec) -> Blueprint` — deterministic running bond. Even courses =
    all full bricks; odd courses = half brick at each end (halfsteensverband).
  - `Blueprint(spec, length, targets, ...)`, `BrickTarget(tid, course, slot, x, y, kind)`,
    `BrickKind{FULL, HALF}`.
  - Suites: `INTERP_SPECS=((5,4),(7,3),(6,5))`, `EXTRAP_SPECS=((9,5),(10,6))`,
    `TRAIN_SPECS = product(range(4,9), range(2,6))`. **Realistic env scale tops out around
    10 modules × 6 courses (~60 bricks).**
- **Consumers to reuse:** `atrium_sim/render/` (renderer + GIF recorder),
  `baselines/oracle.py` (places every target at its pose — the integration test), the envs
  in `atrium_sim/envs/`.
- **Tooling:** `uv` (+ `.venv`). `Pillow` is already a dependency. Repo `.env` (gitignored)
  holds vision-capable keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`,
  `GROQ_API_KEY`, `XAI_API_KEY`.

---

## 3. Design

### 3.1 `FacadePlan` IR

New module `atrium_sim/facade.py` (import-light; dataclasses or pydantic models).
Global grid: `1 col = 1 module (220 mm)`, `1 row = 1 course (60 mm)`, **origin = bottom-left
of the brickwork**.

```python
@dataclass(frozen=True)
class FacadePanel:
    spec: WallSpec          # solid running-bond rectangle
    origin_col: int         # module column of the panel's bottom-left
    origin_row: int         # course row of the panel's bottom-left
    label: str = ""         # e.g. "left pier", "window sill"

@dataclass(frozen=True)
class Opening:
    kind: str               # "window" | "door" | "arched_window" | "patio_door" | ...
    col: int; row: int
    n_cols: int; n_rows: int

@dataclass(frozen=True)
class FacadePlan:
    image_ref: str
    grid_cols: int
    grid_rows: int
    panels: tuple[FacadePanel, ...]
    openings: tuple[Opening, ...]
    notes: str = ""
```

Required helpers:
- `to_json() / from_json()` (round-trippable; this is the VLM's output contract).
- `blueprints() -> list[tuple[Blueprint, tuple[int,int]]]` — one `generate_blueprint(panel.spec)`
  per panel, paired with its `(origin_col, origin_row)` offset.
- `validate()` — panels are within `grid_cols/grid_rows`; panels don't overlap each other;
  panels don't overlap openings; grid is non-degenerate. Raise a clear error otherwise.

**Commit this module (with tests) FIRST on the branch**, before the VLM client, so partial
progress always survives an interruption.

### 3.2 Provider-agnostic VLM client

`vlm/client.py`. One function: `image + instruction → JSON` matching the FacadePlan contract.
- Use the provider's **structured/JSON output** mode; validate with pydantic.
- **Default provider = OpenAI GPT-4o** (matches the project's original spec). `--provider`
  switches to `anthropic` (Claude vision), `google` (Gemini), or `groq` (Llama-Vision).
- Resolve keys from `.env`; prefer the requested provider, else first available; clear error
  if none. Accept local path or URL (download to a temp file; never commit it).
- If the model returns prose around the JSON, extract the JSON block and **retry once** with a
  "return strict JSON only" reminder before failing.

### 3.3 CLI

```
uv run python -m vlm.plan_from_image <image-path-or-url> \
    [--provider openai] [--out plans/<name>.json] [--render]
```
- Writes the `FacadePlan` JSON to `plans/` (create the dir; JSON is fine to commit — no
  copyrighted pixels, just integers).
- `--render` composes all panels (via `blueprints()` + the existing renderer, or a simple
  PIL/matplotlib elevation) into one PNG under `media/` for eyeballing. Draw openings as voids.

### 3.4 VLM prompt (the crux)

Instruct the model to:
1. Focus on the **primary/front elevation**; ignore roof, landscaping, sky.
2. Estimate the overall **brick field** as a `grid_cols × grid_rows` module grid, using
   220 mm × 60 mm modules. Give it scale anchors (typical residential: storey ≈ 2.4–2.7 m,
   door ≈ 0.9 × 2.1 m, brick course ≈ 60 mm) so it can calibrate from visual proportions.
3. Locate every **opening** (windows, doors, patio doors) as a grid rectangle
   `(col,row,n_cols,n_rows)`.
4. Decompose the **remaining brick** into **non-overlapping rectangular running-bond panels**
   (piers between openings, sills below, heads/spandrels above) with origins.
5. Mark non-brick regions (e.g. half-timbered gables) in `notes`, not as panels.
6. Return **STRICT JSON only**, conforming to the FacadePlan schema.

### 3.5 Tests (`tests/`, no network)

- Mock the VLM client to return canned JSON → assert it parses to a valid `FacadePlan`.
- Assert `validate()` catches overlaps / out-of-grid panels.
- Assert `blueprints()` yields valid `Blueprint`s whose brick counts match the panels.
- Keep the full existing suite green: `uv run pytest`.

---

## 4. Acceptance criteria

1. `uv run pytest` green (new offline tests + existing suite).
2. End-to-end run on the **colonial test image** (below) with a real provider if a key works;
   otherwise prove the full path with the mock and **say so**. Test image (do **not** commit it —
   watermarked stock):
   `https://media.istockphoto.com/id/171302239/photo/colonial-house.jpg?s=612x612&w=0&k=20&c=728Aj3qj1UED-twfeCu1IT8eD5rIbSr5fVTda-WwgLM=`
   It's a red-brick Tudor-Revival cottage: big **arched picture window** (left bay), recessed
   **red entry door** (center), **sliding patio door** + a smaller **window** (right wing), two
   chimneys, and a **half-timbered (non-brick) left gable peak**.
3. Produce `plans/colonial.json` and (with `--render`) a composed elevation PNG in `media/`.
4. **Sanity reference** (for comparison, do NOT hardcode): overall field ≈ 55 modules × 44
   courses; ≈ 11 panels around 4 openings; ≈ 1,600 bricks.

---

## 5. The scale gap (call it out, don't hide it)

A real facade is ~55×44 (~1,600 bricks); the env's realistic scale is ≤10×6 (~60 bricks), and
piers are ~44 courses vs the current max of 6. v1 should **record this in `FacadePlan.notes`
and the README** — panels may exceed current training sizes; that's expected. Two later paths:
(a) scale the env (curriculum to taller `C_MAX`, hundreds/thousands of bricks); (b) keep panels
env-sized and compose. A good demo today: any panel that lands near `WallSpec(10,6)` is already
solved by `robot11`, so render that slice laying itself.

---

## 6. Constraints & conventions

- Branch `feat/vlm-facade-plan`. **Commit in small chunks** (schema first). Each commit message
  ends with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Match the repo's clean, single-file-friendly style; type hints; no heavy frameworks.
- Keep `atrium_sim` import-light (no torch/gym in `facade.py`/`blueprint.py`).
- Update `README.md`: tick the VLM-observations roadmap item; add a short "Facade plans from
  images" section with the command and one result image.
- **Do NOT** `git add` the `.env`, the stock image, or any copyrighted media. **Do NOT push or
  open a PR** — leave that to Bruno.

---

## 7. Open questions / follow-ups

- **Arched openings:** v1 can treat the arched picture window as a rectangle; v2 adds brick
  voussoir/soldier courses for the arch head.
- **Chimneys:** the left chimney is ground-up (a tall narrow stack); the right one starts on the
  roof (not lay-from-ground) — exclude from the buildable field or model separately.
- **Gable rake:** triangular brick gables need stepped/cut courses; out of scope for the
  rectangular-panel v1.
- **Multi-elevation:** a photo shows one or two faces; a true "series of 2D walls" (front/side/
  rear) would come from a drawing set. Consider a `FacadePlan` per elevation later.

---

## 8. Ready-to-paste kickoff prompt

> Implement `VLM_FACADE_PLAN_PROPOSAL.md` in this repo. Read it fully, then read
> `atrium_sim/blueprint.py`, `atrium_sim/constants.py`, `atrium_sim/render/`, and
> `baselines/oracle.py` before writing. Work on branch `feat/vlm-facade-plan`, commit the
> `FacadePlan` IR + its tests first, then the VLM client, CLI, rendering, and README. Follow
> the acceptance criteria and constraints in the proposal. Don't push or open a PR; don't
> commit `.env`, the stock image, or copyrighted media. Report branch, files, the exact CLI
> command, whether a real VLM call succeeded (which provider) or only the mock, the colonial
> plan summary (grid, #panels, #openings), artifact paths, and test results.
