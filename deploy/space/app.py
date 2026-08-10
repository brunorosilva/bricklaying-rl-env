"""Entrypoint for the OPTIONAL live backend as a Hugging Face Space - Gradio SDK, not
Docker SDK: creating a Docker-SDK Space now requires HF PRO (a policy change as of ~July
2026), while a Gradio-SDK Space on ZeroGPU hardware remains free to create. HF's Gradio-SDK
runtime just runs this file (`python app.py`), which has to result in a live server - there
is no Dockerfile in this deployment path at all, unlike a Docker-SDK Space.

The actual API (GET /policies, POST /episode - see webviz/api.py) is untouched: this file
only wraps it. It must be grafted onto gradio's OWN app instance *after* a real
`demo.launch()` call, not the other way around (mounting gradio into a hand-built FastAPI
app served via a manual `uvicorn.run()`) - see `_merge_api_routes`'s docstring for why that
first approach silently breaks ZeroGPU's own startup handshake.
"""

import gradio as gr
import spaces

from webviz.api import app as api_app  # the real FastAPI app: GET /policies, POST /episode, etc.


@spaces.GPU
def _zerogpu_probe() -> str:
    """ZeroGPU hardware refuses to start a Space unless at least one @spaces.GPU function
    is wired to a real Gradio event (a bare decorated-but-unused function isn't enough).
    This backend never actually needs a GPU (pure CPU physics sim + small MLP forward
    passes) - this exists solely to satisfy that platform requirement, wired to a hidden
    button below rather than exposed as something a visitor would find useful to click."""
    import torch

    return f"cuda available inside this call: {torch.cuda.is_available()}"


# A light brand pass on Gradio's own chrome - not a full custom theme (this page explicitly
# isn't the product, see the Markdown below), just enough that a visitor arriving here
# straight from HF's Spaces directory sees the same warm-dark/clay identity as the actual
# frontend, not Gradio's cheerful default.
BRAND_CSS = """
.gradio-container { background: #0F0E0D !important; }
.gradio-container, .gradio-container * { color: #EDE9E3; }
a, a:visited { color: #F2B94B !important; }
"""

with gr.Blocks(title="Bricklaying with RL webviz API", css=BRAND_CSS) as demo:
    gr.Markdown(
        "# 🧱 Bricklaying with RL\n\n"
        "**This page isn't the product** - it's the live physics backend behind "
        "[**the Bricklaying with RL frontend**](https://brunorosilva.github.io/bricklaying-rl-env/), "
        "which is where you actually want to be. The featured policy there lays a "
        "16-module wall to **100% fill, 99.4% within ±3&nbsp;mm** of blueprint - real "
        "rigid-body physics underneath, judged like a site inspection.\n\n"
        "See `/docs` for the raw API this Space exposes (`GET /policies`, `POST /episode`)."
    )
    # hidden: exists only so the ZeroGPU platform check above finds a wired @spaces.GPU
    # function; nothing in the real API path calls this.
    _probe_btn = gr.Button(visible=False)
    _probe_out = gr.Textbox(visible=False)
    _probe_btn.click(_zerogpu_probe, outputs=_probe_out)


def _merge_api_routes() -> None:
    """Grafts webviz.api's routes/CORS policy onto gradio's OWN app instance (`demo.app`) -
    the reverse of `gradio.mount_gradio_app`, and only callable *after* `demo.launch()` has
    actually created that instance.

    This has to run after a genuine `demo.launch()` call, not instead of one: the `spaces`
    package's ZeroGPU startup handshake (`spaces.zero.gradio.one_launch`, which monkeypatches
    `gr.Blocks.launch` to run its own registration/reporting step) only fires when
    `Blocks.launch()` is actually invoked. An earlier version of this file used
    `gradio.mount_gradio_app` to embed the FastAPI app inside gr.Blocks and served the
    result with a hand-rolled `uvicorn.run()`, bypassing `.launch()` entirely - that
    deployed, but the Space failed at startup with "No @spaces.GPU function detected during
    startup" even with a wired @spaces.GPU function, because the handshake that reports
    "yes, one exists" never ran.

    Route registration (`include_router`) takes effect immediately - routes are just a list
    Starlette matches per-request. CORS middleware does not: Starlette caches its built
    middleware stack in `app.middleware_stack` on the first ASGI call, and by the time
    `demo.launch()` returns (even with `prevent_thread_lock=True`) that first call has
    already happened internally, so `demo.app.add_middleware(...)` would raise "Cannot add
    middleware after an application has started". Resetting `middleware_stack` to `None`
    forces Starlette to rebuild it - incorporating the middleware just added - on the next
    request.

    Gradio's own `CustomCORSMiddleware` (`strict_cors=True`) is dropped here rather than left
    alongside ours, to avoid two competing CORS layers - but this is *not* what actually
    governs cross-origin access once deployed: confirmed against the live Space that HF's own
    edge unconditionally adds `Access-Control-Allow-Origin: <the request's Origin>` to every
    response from a public `*.hf.space` domain, for *any* path (including ones neither this
    app nor gradio registers) and *regardless* of what the application itself decided - by
    platform design, since Spaces are meant to be callable/embeddable from anywhere. So
    `webviz/api.py`'s `ALLOWED_ORIGINS` restriction has no browser-enforced effect on this
    deployment path; it's kept for the non-Spaces case (plain `uvicorn webviz.api:app` in dev,
    or a future Docker-SDK deployment) where no such edge sits in front of it, and because it
    costs nothing to leave in place. Accepted as a low-severity gap, not chased further: there
    is no sensitive data behind these endpoints, and the protections that actually matter for
    abuse (the policy/spec whitelist, the plan-size clamp, the concurrency semaphore) are all
    application-level and completely unaffected by what the edge does with CORS headers.
    Verified against the live Space: GET /policies and POST /episode (including the plan-size
    clamp) work correctly through the merged routes.
    """
    demo.app.include_router(api_app.router)
    demo.app.user_middleware = api_app.user_middleware + [
        mw for mw in demo.app.user_middleware if "CORS" not in mw.cls.__name__
    ]
    demo.app.middleware_stack = None


if __name__ == "__main__":
    import time

    # ssr_mode=False: Gradio 6's default on Spaces runs a Node.js SSR proxy in front of the
    # Python backend ("Running on local URL: ..., with SSR (Node proxy -> Python :7861)",
    # confirmed in this Space's own boot log) - it only forwards paths Gradio's own
    # frontend knows about, so /policies and /episode never reached _merge_api_routes'
    # grafted routes at all; they got the SPA shell instead, silently, no error anywhere.
    # Without SSR, Python serves everything directly on the port HF actually proxies to.
    demo.launch(server_name="0.0.0.0", prevent_thread_lock=True, ssr_mode=False)
    _merge_api_routes()
    while True:
        time.sleep(3600)
