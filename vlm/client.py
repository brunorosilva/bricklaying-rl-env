"""Gemini facade perception: an image -> a module grid + openings (structured JSON).

Perception only. The model estimates the brick field as a grid of 220mm x 60mm modules
and locates the openings; the deterministic tiler in atrium_sim/facade.py does the
geometry. We ask for an approximate-but-cohesive read, not metric truth.
"""

from __future__ import annotations

import mimetypes
import urllib.request

from pydantic import BaseModel, Field


class OpeningOut(BaseModel):
    kind: str = Field(description="window | door | patio_door | arched_window | garage | ...")
    col: int = Field(description="module column of the opening's bottom-left (0 = left edge)")
    row: int = Field(description="course row of the opening's bottom-left (0 = ground)")
    n_cols: int = Field(description="width in modules")
    n_rows: int = Field(description="height in courses")


class PerceptionOut(BaseModel):
    grid_cols: int = Field(description="total brick field width in 220mm modules")
    grid_rows: int = Field(description="total brick field height in 60mm courses")
    openings: list[OpeningOut]
    notes: str = Field(default="", description="non-brick regions (gables, timber, roof)")


PROMPT = """You are reading the FRONT ELEVATION of a brick building to produce a buildable
brick plan. Model the brickwork as a grid of modules: 1 column = one 220mm brick module,
1 row = one 60mm brick course. The origin (col 0, row 0) is the BOTTOM-LEFT of the brickwork.

Calibrate scale from visual proportions using these anchors: a storey is ~2.4-2.7m (~40-45
courses), a standard door is ~0.9m wide x ~2.1m tall (~4 modules x ~35 courses), a window
is ~1.0-1.5m wide. Ignore roof, sky, landscaping, chimneys above the roofline, and any
non-brick material (timber, render, stone) - mention those in `notes`, do not treat them
as openings.

Return:
- grid_cols, grid_rows: the overall brick field size in modules x courses.
- openings: every window/door/patio-door as a rectangle (col,row,n_cols,n_rows) in grid
  coordinates, bottom-left origin. Be roughly proportionate; approximate is fine, but the
  openings must sit inside the grid and not overlap each other.
- notes: any non-brick regions.

Approximate but COHESIVE - the openings and grid should look right relative to each other."""


def _load_image(src: str) -> tuple[bytes, str]:
    """Return (bytes, mime) for a local path or URL. URL is downloaded in-memory (never
    written to the repo - it may be copyrighted stock)."""
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            mime = r.headers.get_content_type() or "image/jpeg"
        return data, mime
    with open(src, "rb") as f:
        data = f.read()
    return data, mimetypes.guess_type(src)[0] or "image/jpeg"


def perceive_facade(image_src: str, api_key: str,
                    model: str = "gemini-2.5-flash") -> tuple[str, PerceptionOut]:
    """One vision call: image -> (raw JSON text, parsed PerceptionOut)."""
    from google import genai
    from google.genai import types

    img_bytes, mime = _load_image(image_src)
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime), PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PerceptionOut,
            temperature=0.2,
        ),
    )
    raw = resp.text
    parsed = resp.parsed if getattr(resp, "parsed", None) else PerceptionOut.model_validate_json(raw)
    return raw, parsed
