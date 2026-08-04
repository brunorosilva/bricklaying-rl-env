"""image -> Gemini perception -> FacadePlan (+ elevation render).

    uv run python -m vlm.plan_from_image <image-url-or-path> --name colonial --render

Saves plans/<name>.json (the plan) and plans/<name>_vlm_raw.json (the raw model
response - "the process", so it's reproducible without another call). With --render,
composes all panels + openings into media/<name>_facade.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from atrium_sim.constants import COURSE_MM, MODULE_MM
from atrium_sim.facade import FacadePlan, Opening


def _load_key(name: str = "GOOGLE_API_KEY") -> str | None:
    import os

    if os.environ.get(name):
        return os.environ[name]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def render_facade(plan: FacadePlan, path: str, px_per_mm: float = 0.16) -> None:
    """Compose every panel's bricks (running bond, quality-neutral) + openings (voids)
    into one elevation PNG, using PIL. Origin bottom-left; y flips for image space."""
    from PIL import Image, ImageDraw

    from atrium_sim.blueprint import brick_face

    W = int(plan.grid_cols * MODULE_MM * px_per_mm) + 20
    H = int(plan.grid_rows * COURSE_MM * px_per_mm) + 20
    img = Image.new("RGB", (W, H), (28, 30, 36))
    d = ImageDraw.Draw(img)
    field_h = plan.grid_rows * COURSE_MM

    def to_px(x_mm, y_mm):  # bottom-left origin -> top-left image
        return 10 + x_mm * px_per_mm, 10 + (field_h - y_mm) * px_per_mm

    for bp, (ocol, orow) in plan.blueprints():
        ox, oy = ocol * MODULE_MM, orow * COURSE_MM
        for t in bp.targets:
            w, h = brick_face(t.kind)
            x0, y0 = to_px(ox + t.x - w / 2, oy + t.y + h / 2)
            x1, y1 = to_px(ox + t.x + w / 2, oy + t.y - h / 2)
            d.rectangle([x0, y0, x1, y1], fill=(178, 92, 62), outline=(120, 70, 50))
    for o in plan.openings:
        x0, y0 = to_px(o.col * MODULE_MM, (o.row + o.n_rows) * COURSE_MM)
        x1, y1 = to_px((o.col + o.n_cols) * MODULE_MM, o.row * COURSE_MM)
        d.rectangle([x0, y0, x1, y1], fill=(18, 20, 26), outline=(90, 130, 170))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="local path or URL of the facade image")
    p.add_argument("--name", default="facade")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--render", action="store_true")
    p.add_argument("--mock", help="skip the VLM call; load perception JSON from this file")
    a = p.parse_args()

    if a.mock:
        raw = Path(a.mock).read_text()
        from vlm.client import PerceptionOut
        perc = PerceptionOut.model_validate_json(raw)
    else:
        key = _load_key()
        if not key:
            sys.exit("GOOGLE_API_KEY not found (env or .env)")
        from vlm.client import perceive_facade
        print(f"[vlm] calling {a.model} on {a.image} ...", flush=True)
        raw, perc = perceive_facade(a.image, key, model=a.model)

    Path("plans").mkdir(exist_ok=True)
    Path(f"plans/{a.name}_vlm_raw.json").write_text(raw)

    plan = FacadePlan.from_perception(
        a.image, perc.grid_cols, perc.grid_rows,
        [Opening(o.kind, o.col, o.row, o.n_cols, o.n_rows) for o in perc.openings],
        notes=perc.notes,
    )
    Path(f"plans/{a.name}.json").write_text(plan.to_json())

    print(f"[vlm] grid {plan.grid_cols}x{plan.grid_rows}  openings={len(plan.openings)}  "
          f"panels={len(plan.panels)}  bricks={plan.n_bricks}")
    print(f"[vlm] wrote plans/{a.name}.json + plans/{a.name}_vlm_raw.json")
    if a.render:
        out = f"media/{a.name}_facade.png"
        render_facade(plan, out)
        print(f"[vlm] rendered {out}")


if __name__ == "__main__":
    main()
