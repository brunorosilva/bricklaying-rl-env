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


def list_checkpoints() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    # exclude robot runs (kept in runs/robot); those use the hybrid loader
    return sorted(str(p.parent.name) for p in RUNS_DIR.glob("*/ckpt.pt"))


def list_robot_checkpoints() -> list[str]:
    if not ROBOT_RUNS_DIR.exists():
        return []
    return sorted(str(p.parent.name) for p in ROBOT_RUNS_DIR.glob("*/ckpt.pt"))


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

        return HybridAgentPolicy(load_hybrid_agent(str(ROBOT_RUNS_DIR / name[5:] / "ckpt.pt")))
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


def _robot_drop_control(policy_name: str) -> bool:
    """A drop-trained checkpoint stores drop_control in its args; the replay env must
    match so the release-height mechanic reproduces (older checkpoints default False)."""
    if not policy_name.startswith("ckpt:"):
        return False
    try:
        import torch

        ck = torch.load(str(ROBOT_RUNS_DIR / policy_name[5:] / "ckpt.pt"),
                        weights_only=True, map_location="cpu")
        return bool(ck.get("args", {}).get("drop_control", False))
    except Exception:
        return False


def run_robot_episode(policy_name: str, seed: int, spec_str: str,
                      scenario: str = "empty") -> dict:
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    u = env.unwrapped
    drop = _robot_drop_control(policy_name)          # match the checkpoint's mechanic
    prefill = 1.0 if scenario == "prefill" else 0.0  # force a random partial structure to complete
    if drop or prefill:
        u.env_cfg = type(u.env_cfg)(drop_control=drop, prefill_prob=prefill)
    try:
        policy = build_robot_policy(policy_name, env)
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
