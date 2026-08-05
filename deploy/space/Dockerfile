# Builds the OPTIONAL live backend (webviz/api.py) as a Hugging Face Space (Docker SDK).
# Not required for the site to work - see the repo README for how the static export +
# precomputed trace matrix (scripts/export_traces.py) covers the home page on its own.
# This exists for the /build grid editor and arbitrary-seed replays.
#
# .github/workflows/space.yml stages a subtree containing exactly what's copied below
# (atrium_sim/, webviz/, train/, baselines/, plans/, runs/**/ckpt.pt, pyproject.toml,
# uv.lock) plus this file (as ./Dockerfile) and deploy/space/README.md (as ./README.md,
# whose YAML frontmatter IS the Space's config card) and pushes it to the Space's git
# remote - see that workflow and deploy/space/README.md for the one-time setup.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Created before the first COPY so `COPY --chown=user` can resolve the name (BuildKit
# looks it up in the image's /etc/passwd at COPY time). uid 1000 is what a Space's
# container actually runs as.
RUN useradd -m -u 1000 user

WORKDIR /app

# No build-essential: every package in the resolved serve+infer set publishes a cp311
# manylinux wheel (checked pymunk, cffi, torch, numpy, pydantic-core, uvloop, watchfiles,
# httptools, websockets, pillow, grpcio individually against uv.lock - nothing here is
# ever compiled), so a ~250MB toolchain layer would be pure overhead. --only-binary=:all:
# keeps that invariant enforced: a future lockfile bump that introduces an sdist-only dep
# fails the build with "no wheel available" instead of silently needing the toolchain back.
#
# Resolve third-party deps from the committed lockfile (reproducible - matters here since
# atrium_sim/physics.py's determinism guarantee is per-platform/per-pymunk-build; pinning
# keeps a live replay from silently diverging from what scripts/export_traces.py baked on
# CI). `--locked` fails the build if uv.lock is stale relative to pyproject.toml rather than
# silently re-resolving inside the container. `--no-emit-project` excludes atrium_sim itself
# - installed properly (non-editable) below once its source is actually present. `serve` is
# fastapi+uvicorn, `infer` is torch alone: no tensorboard/tyro/wandb/matplotlib/pygame. uv
# itself is removed in the same layer so it doesn't ship in the image.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv export --extra serve --extra infer --no-dev --no-hashes --no-emit-project \
        --emit-index-url --locked --format requirements-txt -o requirements.txt && \
    pip install --no-cache-dir --only-binary=:all: -r requirements.txt && \
    pip uninstall -y -q uv && rm requirements.txt

# Only what webviz/api.py's import chain actually touches - no render/plot/track extras,
# no tests, no frontend, no non-ckpt run artifacts (gifs, tfevents logs - the Space subtree
# staged by .github/workflows/space.yml only ever contains runs/**/ckpt.pt in the first
# place).
#
# --chown=user per COPY rather than a trailing `RUN chown -R user:user /app`: a recursive
# chown rewrites metadata for every file, which Docker stores as a full second copy in a new
# layer (HF's own Docker Spaces docs call this out by name).
#
# webviz/, train/, baselines/ are plain directories on PYTHONPATH (they aren't declared
# wheel packages - see pyproject.toml's [tool.hatch.build.targets.wheel] - and never have
# been; that's also how `uv run python -m webviz.episode` resolves them from repo root in
# dev). atrium_sim IS a real package, so it's also pip-installed properly here - belt and
# braces alongside PYTHONPATH, in case a future uvicorn/base-image change stops treating
# /app as import root.
#
# README.md is here because pyproject.toml's `readme = "README.md"` is resolved by
# hatchling when it builds the wheel below - COPY is selective, so without this line the
# file is never in the image (regardless of whether the build CONTEXT has one) and the
# build dies with "Readme file does not exist". In the Space subtree that file is
# deploy/space/README.md (staged as ./README.md); for a local
# `docker build -f deploy/space/Dockerfile .` it's the repo's own README.
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user atrium_sim ./atrium_sim
COPY --chown=user webviz ./webviz
COPY --chown=user train ./train
COPY --chown=user baselines ./baselines
COPY --chown=user plans ./plans
COPY --chown=user runs ./runs
RUN pip install --no-cache-dir --no-deps .

# site-packages was installed system-wide as root above, so this is just the runtime
# identity, not a permissions workaround.
USER user

# Fail the BUILD, not the Space, if the import chain or file ownership is wrong - a
# RUNTIME_ERROR on HF is much harder to read than a red build log. Runs as `user`, so it
# also proves the runtime identity can actually read everything COPYed above.
RUN python -c "import webviz.api"

EXPOSE 7860
CMD ["uvicorn", "webviz.api:app", "--host", "0.0.0.0", "--port", "7860"]
