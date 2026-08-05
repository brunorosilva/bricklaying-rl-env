# Builds the OPTIONAL live backend (webviz/api.py) as a Hugging Face Space (Docker SDK).
# Not required for the site to work - see frontend/README or the repo README for how the
# static export + precomputed trace matrix (scripts/export_traces.py) covers the home page
# on its own. This exists for the /build grid editor and arbitrary-seed replays.
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

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Resolve third-party deps from the committed lockfile (reproducible - matters here since
# atrium_sim/physics.py's determinism guarantee is per-platform/per-pymunk-build; pinning
# keeps a live replay from silently diverging from what scripts/export_traces.py baked on
# CI). `--no-emit-project` excludes atrium_sim itself - installed properly (non-editable)
# below once its source is actually present, rather than as an `-e .` pointing nowhere yet.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv export --extra serve --extra train --no-dev --no-hashes --no-emit-project --emit-index-url \
        --format requirements-txt -o requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Only what webviz/api.py's import chain actually touches - no render/plot/track extras,
# no tests, no frontend, no non-ckpt run artifacts (gifs, tfevents logs: see .dockerignore).
#
# webviz/, train/, baselines/ are plain directories on PYTHONPATH (they aren't declared
# wheel packages - see pyproject.toml's [tool.hatch.build.targets.wheel] - and never have
# been; that's also how `uv run python -m webviz.episode` resolves them from repo root in
# dev). atrium_sim IS a real package, so it's also pip-installed properly here - belt and
# braces alongside PYTHONPATH, in case a future uvicorn/base-image change stops treating
# /app as import root.
COPY atrium_sim ./atrium_sim
COPY webviz ./webviz
COPY train ./train
COPY baselines ./baselines
COPY plans ./plans
COPY runs ./runs
RUN pip install --no-cache-dir --no-deps .

# HF Spaces run the container as a non-root user by convention; site-packages was already
# installed system-wide as root above, so this is just the runtime identity, not a
# permissions workaround.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "webviz.api:app", "--host", "0.0.0.0", "--port", "7860"]
