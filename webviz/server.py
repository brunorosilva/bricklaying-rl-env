"""Zero-dependency web UI to watch a policy lay a wall.

    uv run python -m webviz.server                 # http://0.0.0.0:8000  (reach over Tailscale)
    uv run python -m webviz.server --port 8000 --ngrok

The server binds 0.0.0.0 so any device on your tailnet can open it at
http://<your-tailscale-ip>:<port>. It runs an episode on demand for a chosen
policy (random / greedy / oracle / any trained checkpoint under runs/) and
streams back a compact replay the browser animates on a canvas.

--ngrok is best-effort: it needs the `ngrok` binary + an authtoken already
configured (`ngrok config add-authtoken ...`). Over Tailscale you usually
don't need it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import gymnasium as gym

import atrium_sim  # noqa: F401  (registers the env)
from atrium_sim.blueprint import WallSpec
from webviz.trajectory import record_robot_trajectory, record_trajectory

HERE = Path(__file__).parent
INDEX = HERE / "index.html"
RUNS_DIR = Path("runs")
ROBOT_RUNS_DIR = Path("runs/robot")
PLANS_DIR = Path("plans")
HOUSE_PREFIX = "house:"


def _robot_search_roots() -> list[Path]:
    """runs/robot (the curated/shipped location) plus any architecture-bake-off directory
    (train.sweep's --run-dir, e.g. runs/sweep_archbakeoff*) - so a sweep run's checkpoints
    are inspectable in the frontend without first being promoted into runs/robot. Safe to
    widen: list_robot_checkpoints' obs_dim compat check still filters out anything that
    isn't a current-env robot checkpoint (old grid-obs sweeps, base-task sweeps, ...)."""
    roots = [ROBOT_RUNS_DIR]
    roots += sorted(p for p in Path("runs").glob("*sweep*") if p.is_dir() and p != ROBOT_RUNS_DIR)
    return roots


def _find_robot_run_dir(name: str) -> Path:
    for root in _robot_search_roots():
        d = root / name
        if (d / "ckpt.pt").exists():
            return d
    raise FileNotFoundError(f"no robot checkpoint run {name!r} under {_robot_search_roots()}")


def list_checkpoints() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    # exclude robot runs (kept in runs/robot); those use the hybrid loader
    return sorted(str(p.parent.name) for p in RUNS_DIR.glob("*/ckpt.pt"))


_robot_ckpt_compat_cache: dict[str, tuple[float, bool]] = {}  # path -> (mtime, is_compatible)


def list_robot_checkpoints() -> list[str]:
    """Only checkpoints whose obs_dim matches the CURRENT robot env (the sensor obs).
    Older grid-obs checkpoints (robot9-14) can't run against the new env, so hide them
    rather than let a selection error out.

    Cached by (path, mtime): torch.load-ing every checkpoint under runs/robot/ on EVERY
    call (confirmed: this endpoint is hit on every frontend page load) was the dominant
    latency cost of opening the page - a stat() per file is orders of magnitude cheaper,
    and a checkpoint overwritten mid-training (mtime changes) still gets re-checked."""
    import torch

    from atrium_sim.envs.robot_env import OBS_DIM
    out = []
    paths = [p for root in _robot_search_roots() if root.exists()
             for p in root.glob("*/ckpt.pt")]
    for p in sorted(paths, key=lambda p: p.parent.name):
        key = str(p)
        mtime = p.stat().st_mtime
        cached = _robot_ckpt_compat_cache.get(key)
        if cached is not None and cached[0] == mtime:
            compatible = cached[1]
        else:
            try:
                ck = torch.load(str(p), weights_only=True, map_location="cpu")
                compatible = int(ck.get("obs_dim", -1)) == OBS_DIM
            except Exception:
                compatible = False
            _robot_ckpt_compat_cache[key] = (mtime, compatible)
        if compatible:
            out.append(p.parent.name)
    return out


def build_policy(name: str, env):
    """name is 'random' | 'greedy' | 'oracle' | 'ckpt:<run-dir-name>'."""
    if name.startswith("ckpt:"):
        from train.agent import CheckpointPolicy

        return CheckpointPolicy(str(RUNS_DIR / name[5:] / "ckpt.pt"))
    from baselines.policy import make_policy

    return make_policy(name, env, seed=0)


def parse_spec(spec_str: str) -> WallSpec | None:
    if not spec_str or spec_str == "random":
        return None
    m, c = spec_str.lower().split("x")
    return WallSpec(int(m), int(c))


def run_episode(policy_name: str, seed: int, spec_str: str, scenario: str = "empty") -> dict:
    env = gym.make("atrium_sim/BrickLayer-v0")
    try:
        policy = build_policy(policy_name, env)
        return record_trajectory(
            env, policy, seed=seed, spec=parse_spec(spec_str), scenario=scenario
        )
    finally:
        env.close()


def build_robot_policy(name: str, env):
    """Robot policies: 'oracle' | 'random' | 'ckpt:<robot-run-dir>' (hybrid)."""
    if name.startswith("ckpt:"):
        from train.agent import HybridAgentPolicy, load_hybrid_agent

        return HybridAgentPolicy(load_hybrid_agent(str(_find_robot_run_dir(name[5:]) / "ckpt.pt")))
    if name == "oracle":
        from baselines.robot_oracle import RobotOraclePolicy

        return RobotOraclePolicy(env)
    if name == "random":
        space = env.action_space

        class _Rand:
            def act(self, obs):
                return space.sample()

        space.seed(0)
        return _Rand()
    raise ValueError(f"unknown robot policy: {name!r}")


def _robot_ckpt_overrides(policy_name: str) -> tuple[dict, dict]:
    """(env_cfg overrides, reward_cfg overrides) the checkpoint was trained with, so replay
    reproduces both the MECHANICS (drop-height release, rail-edge fall) and the REWARD SCALE
    (sigma_mm/deg, collapse_penalty, c_waste) - train.ppo_robot.make_env applies all of these
    on top of the env's own defaults, so without them replay's displayed reward/return numbers
    are on a different scale than what training actually optimized. Older checkpoints (no
    saved args, or missing fields) default to the pre-training-override behavior rather than
    erroring."""
    if not policy_name.startswith("ckpt:"):
        return {}, {}
    try:
        import torch

        ck = torch.load(str(_find_robot_run_dir(policy_name[5:]) / "ckpt.pt"),
                        weights_only=True, map_location="cpu")
        a = ck.get("args", {})
        env_overrides = {
            "drop_control": bool(a.get("drop_control", False)),
            "fall_off_edge": bool(a.get("fall_off_edge", False)),
        }
        reward_overrides = {
            "sigma_mm": float(a.get("sigma_mm", 6.0)), "sigma_deg": float(a.get("sigma_deg", 2.0)),
            "collapse_penalty": 0.5, "c_waste": 0.25,  # make_env's own fixed overrides
        }
        return env_overrides, reward_overrides
    except Exception:
        return {}, {}


def list_house_plans() -> list[str]:
    """Saved FacadePlans under plans/ that parse as buildable facades (image -> house)."""
    if not PLANS_DIR.exists():
        return []
    from atrium_sim.facade import FacadePlan
    out = []
    for p in sorted(PLANS_DIR.glob("*.json")):
        try:
            if FacadePlan.from_json(p.read_text()).panels:
                out.append(f"{HOUSE_PREFIX}{p.stem}")
        except Exception:
            pass
    return out


def _load_house_plan(spec_str: str):
    """A FacadePlan if spec_str is 'house:<name>' (-> plans/<name>.json), else None.

    `name` must resolve to a plain file directly inside PLANS_DIR - rejects any path
    component (a slash) and any resolved path that escapes PLANS_DIR (`..`, a symlink,
    an absolute path smuggled in as `name`), so `house:../../etc/passwd`-style spec
    strings can't be used to read files outside plans/."""
    if not spec_str.startswith(HOUSE_PREFIX):
        return None
    from atrium_sim.facade import FacadePlan

    name = spec_str[len(HOUSE_PREFIX):]
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"invalid plan name: {name!r}")
    plans_dir = PLANS_DIR.resolve()
    path = (plans_dir / f"{name}.json").resolve()
    if plans_dir not in path.parents:
        raise ValueError(f"invalid plan name: {name!r}")
    return FacadePlan.from_json(path.read_text())


def run_robot_episode(policy_name: str, seed: int, spec_str: str,
                      scenario: str = "empty", plan_json: str | None = None) -> dict:
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    u = env.unwrapped
    env_overrides, reward_overrides = _robot_ckpt_overrides(policy_name)  # match the checkpoint
    prefill = 1.0 if scenario == "prefill" else 0.0  # force a random partial structure to complete
    if plan_json is not None:
        # a custom plan (the grid editor) - too big/structured for --spec's argv round-trip,
        # so it arrives over stdin instead (see webviz.episode --plan-stdin). Validated, not
        # oracle-checked: a plan can be well-formed and still turn out physically unbuildable
        # (see README) - the replay itself is the feedback, same as any other level.
        from atrium_sim.facade import FacadePlan, Opening

        data = json.loads(plan_json)
        if "panels" in data:
            plan = FacadePlan.from_json(plan_json)
        else:
            # untiled: grid_cols/grid_rows/openings only, straight from the grid editor - the
            # browser never computes panels itself, the deterministic tiler does (the same
            # path plans/*.json go through when authored from a photo, see facade.py).
            openings = [Opening(**o) for o in data.get("openings", [])]
            plan = FacadePlan.from_perception(
                data.get("image_ref", "custom"), int(data["grid_cols"]), int(data["grid_rows"]),
                openings, notes=data.get("notes", ""),
            )
        plan.validate()
    else:
        plan = _load_house_plan(spec_str)  # a whole facade/house instead of a single wall
    if env_overrides or prefill:
        u.env_cfg = type(u.env_cfg)(prefill_prob=prefill, **env_overrides)
    if reward_overrides:
        u.reward_cfg = type(u.reward_cfg)(**reward_overrides)
    try:
        policy = build_robot_policy(policy_name, env)
        if plan is not None:
            return record_robot_trajectory(env, policy, seed=seed, plan=plan)
        return record_robot_trajectory(env, policy, seed=seed, spec=parse_spec(spec_str))
    finally:
        env.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter console
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/policies":
            policies = ["oracle", "greedy", "random"] + [f"ckpt:{c}" for c in list_checkpoints()]
            specs = ["random", "4x2", "4x4", "5x4", "6x5", "7x3", "8x5", "10x6"]
            self._json({"policies": policies, "specs": specs})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/episode":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            traj = run_episode(
                req.get("policy", "oracle"),
                int(req.get("seed", 0)),
                req.get("spec", "random"),
            )
            self._json(traj)
        except Exception as e:  # surface the error to the browser instead of a blank 500
            import traceback

            self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}, 500)


def start_ngrok(port: int) -> str | None:
    try:
        subprocess.Popen(
            ["ngrok", "http", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("  ngrok binary not found - install it and run `ngrok config add-authtoken <t>`.")
        print("  (Over Tailscale you can skip ngrok and use the tailnet URL below.)")
        return None
    for _ in range(20):
        time.sleep(0.5)
        try:
            data = json.loads(
                urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1).read()
            )
            for t in data.get("tunnels", []):
                if t.get("public_url", "").startswith("https"):
                    return t["public_url"]
        except Exception:
            continue
    print("  ngrok started but no public URL yet - check http://127.0.0.1:4040")
    return None


def tailscale_ip() -> str | None:
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=2)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ngrok", action="store_true", help="also expose via ngrok (best-effort)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"atrium-sim webviz serving on http://{args.host}:{args.port}")
    ts = tailscale_ip()
    if ts:
        print(f"  tailnet URL:  http://{ts}:{args.port}   <- open this from your laptop")
    if args.ngrok:
        url = start_ngrok(args.port)
        if url:
            print(f"  ngrok URL:    {url}")
    print("  Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        server.shutdown()


if __name__ == "__main__":
    main()
