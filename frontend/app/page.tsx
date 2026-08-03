"use client";

import { useEffect, useRef, useState } from "react";

// ---- types (mirror webviz/trajectory.py output) ----
type Brick = [number, number, number, number]; // x, y, theta(rad), kind(0=full,1=half)
type Target = { x: number; y: number; w: number; h: number; kind: number; course: number; slot: number };
type Step = {
  i: number; reward: number; return: number; cursor?: number;
  frac_in_tol: number; frac_filled: number; waste?: number; ticks: Brick[][];
  mode?: number; base_ticks?: number[]; moves?: number; placements?: number;
};
type Metrics = {
  frac_in_tol: number; frac_filled: number; waste_count: number;
  episode_return: number; mean_abs_dev_mm: number; placements: number; moves?: number;
};
type HardBody = { kind: string; appear: number; verts: [number, number][] };
type Replay = {
  spec: { n_modules: number; n_courses: number };
  length: number; n_courses: number; n_targets: number;
  targets: Target[]; steps: Step[]; metrics: Metrics; seed: number; _policy?: string;
  robot?: { reach: number }; hard_bodies?: HardBody[];
};
type Frame = { st: Step; bricks: Brick[]; base?: number; gi: number };
type View = { xmin: number; ymax: number; s: number; ox: number; oy: number };

// ---- geometry / matching (mirrors atrium_sim reward + renderer) ----
const GATE = 55, TOL = 3, TOL_DEG = 0.5, GATE_DEG = 15;
const MARGIN = 150, HUD_H = 64;

function foldDeg(rad: number): number {
  const t = ((((rad + Math.PI / 2) % Math.PI) + Math.PI) % Math.PI) - Math.PI / 2;
  return Math.abs(t) * 180 / Math.PI;
}

function matchBricks(bricks: Brick[], targets: Target[]): ({ ti: number; d: number } | null)[] {
  const consumed = new Set<number>();
  const matchOf: ({ ti: number; d: number } | null)[] = bricks.map(() => null);
  for (let ti = 0; ti < targets.length; ti++) {
    const t = targets[ti];
    let best = -1, bd = Infinity;
    for (let bi = 0; bi < bricks.length; bi++) {
      if (consumed.has(bi)) continue;
      const b = bricks[bi];
      if (b[3] !== t.kind) continue;
      if (foldDeg(b[2]) > GATE_DEG) continue;
      const d = Math.hypot(b[0] - t.x, b[1] - t.y);
      if (d > GATE) continue;
      if (d < bd) { bd = d; best = bi; }
    }
    if (best >= 0) { consumed.add(best); matchOf[best] = { ti, d: bd }; }
  }
  return matchOf;
}

function qualityColor(d: number, deg: number): string {
  if (d <= TOL && deg <= TOL_DEG) return "#5fb45a";
  const t = Math.min(1, (d - TOL) / 15);
  return `rgb(${Math.round(200 + 30 * t)},${Math.round(150 * (1 - t) + 30)},40)`;
}

function makeView(len: number, nCourses: number, W: number, H: number): View {
  const xmin = -MARGIN, xmax = len + MARGIN, ymin = -30, ymax = nCourses * 60 + 170;
  const worldW = xmax - xmin, worldH = ymax - ymin;
  // the HUD occupies the top HUD_H px, and py() offsets content down by it, so fit the
  // scene into the REMAINING height - otherwise tall walls (facade panels) crop at the bottom
  const availH = H - HUD_H;
  const s = Math.min(W / worldW, availH / worldH);
  return { xmin, ymax, s, ox: (W - worldW * s) / 2, oy: (availH - worldH * s) / 2 };
}
const px = (v: View, x: number) => v.ox + (x - v.xmin) * v.s;
const py = (v: View, y: number) => HUD_H + v.oy + (v.ymax - y) * v.s;

function corners(v: View, cx: number, cy: number, w: number, h: number, theta: number) {
  const c = Math.cos(theta), s = Math.sin(theta);
  return ([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]] as const).map(
    ([dx, dy]) => [px(v, cx + dx * c - dy * s), py(v, cy + dx * s + dy * c)] as [number, number],
  );
}

function poly(ctx: CanvasRenderingContext2D, pts: [number, number][], fill?: string, stroke?: string, dash = false) {
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) {
    ctx.strokeStyle = stroke; ctx.lineWidth = dash ? 1 : 2;
    if (dash) ctx.setLineDash([5, 4]);
    ctx.stroke(); ctx.setLineDash([]);
  }
}

function drawScene(canvas: HTMLCanvasElement, v: View, replay: Replay, frame: Frame, labels: boolean) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = "#0f1116"; ctx.fillRect(0, 0, W, H);

  // ground
  ctx.fillStyle = "#3e434c";
  ctx.fillRect(0, py(v, 0), W, H - py(v, 0));

  // mobile robot: shaded reach window + base marker sliding along the rail
  if (replay.robot && frame.base != null) {
    const reach = replay.robot.reach, gy = py(v, 0);
    const lx = px(v, frame.base - reach), rx = px(v, frame.base + reach);
    ctx.fillStyle = "rgba(237,177,0,0.10)";
    ctx.fillRect(lx, gy, rx - lx, H - gy);
    const bx = px(v, frame.base);
    ctx.fillStyle = "#edb100";
    ctx.beginPath(); ctx.moveTo(bx, gy - 18); ctx.lineTo(bx - 10, gy); ctx.lineTo(bx + 10, gy);
    ctx.closePath(); ctx.fill();
  }

  const matchOf = matchBricks(frame.bricks, replay.targets);
  const filled = new Set<number>();
  matchOf.forEach((m) => m && filled.add(m.ti));

  // ghost targets (unfilled)
  for (let ti = 0; ti < replay.targets.length; ti++) {
    if (filled.has(ti)) continue;
    const t = replay.targets[ti];
    poly(ctx, corners(v, t.x, t.y, t.w, t.h, 0), undefined, "#6e737d", true);
  }

  // static hard bodies (cement arches, lintels, sills) - faded in as the build reaches them
  for (const hb of replay.hard_bodies ?? []) {
    if (hb.appear > frame.gi) continue;
    const pts = hb.verts.map(([x, y]) => [px(v, x), py(v, y)] as [number, number]);
    const fill = hb.kind === "voussoir" ? "#b25c3e" : hb.kind === "cement" ? "#969690" : "#b2aca0";
    poly(ctx, pts, fill, hb.kind === "voussoir" ? "#6a655f" : "#6e6c68");
  }

  // bricks
  ctx.font = `${Math.max(9, 11 * v.s * 2)}px ui-monospace, monospace`;
  ctx.textAlign = "center";
  frame.bricks.forEach((b, bi) => {
    const m = matchOf[bi];
    const w = b[3] === 1 ? 100 : 210, h = 50;
    let fill = "#5a2d28"; // stray / toppled
    if (m) fill = qualityColor(m.d, foldDeg(b[2]));
    else if (foldDeg(b[2]) < GATE_DEG) fill = "#b25c3e"; // upright, in flight
    poly(ctx, corners(v, b[0], b[1], w + 9, h + 9, b[2]), "#6a655f");
    poly(ctx, corners(v, b[0], b[1], w, h, b[2]), fill);
    if (labels && m) {
      const t = replay.targets[m.ti];
      ctx.fillStyle = "#cfd3da";
      ctx.fillText((b[0] - t.x >= 0 ? "+" : "") + (b[0] - t.x).toFixed(1), px(v, b[0]), py(v, b[1] + h / 2 + 14));
    }
  });

  // HUD
  ctx.fillStyle = "#12141a"; ctx.fillRect(0, 0, W, HUD_H);
  const st = frame.st;
  ctx.textAlign = "left"; ctx.font = "13px ui-monospace, monospace";
  ctx.fillStyle = "#edb100"; ctx.fillText("atrium-sim", 12, 20);
  ctx.fillStyle = "#e7e8ec";
  const tail = replay.robot
    ? `moves ${st.moves ?? 0}   placed ${st.placements ?? 0}`
    : `waste ${st.waste ?? 0}`;
  const mode = replay.robot ? ["place", "move ←", "move →"][st.mode ?? 0] + "   " : "";
  ctx.fillText(
    `${replay._policy ?? ""}   ${replay.spec.n_modules}m × ${replay.spec.n_courses}c   ` +
    `step ${st.i + 1}/${replay.steps.length}   ${mode}in-tol ${(st.frac_in_tol * 100).toFixed(0)}%   ` +
    `filled ${(st.frac_filled * 100).toFixed(0)}%   ${tail}   return ${st.return.toFixed(2)}`,
    12, 44,
  );
}

function drawStrip(canvas: HTMLCanvasElement, replay: Replay, curStep: number) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const steps = replay.steps, n = steps.length, bw = W / n, mid = H * 0.55;
  const maxAbs = Math.max(0.5, ...steps.map((s) => Math.abs(s.reward)));
  ctx.strokeStyle = "#2c313c"; ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();
  for (let i = 0; i < n; i++) {
    const r = steps[i].reward, hgt = (Math.abs(r) / maxAbs) * (H * 0.42);
    ctx.fillStyle = i === curStep ? "#edb100" : r >= 0 ? "#5fb45a" : "#d64b45";
    ctx.fillRect(i * bw + 0.5, r >= 0 ? mid - hgt : mid, Math.max(1, bw - 1), hgt);
  }
}

export default function Page() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stripRef = useRef<HTMLCanvasElement>(null);
  const scrubRef = useRef<HTMLInputElement>(null);
  const frameLabelRef = useRef<HTMLSpanElement>(null);

  const replayRef = useRef<Replay | null>(null);
  const tlRef = useRef<Frame[]>([]);
  const curRef = useRef(0);
  const playingRef = useRef(false);
  const speedRef = useRef(1);
  const labelsRef = useRef(false);
  const viewRef = useRef<View | null>(null);

  const [env, setEnv] = useState<"bricklayer" | "robot">("bricklayer");
  const [policies, setPolicies] = useState<string[]>([]);
  const [specs, setSpecs] = useState<string[]>([]);
  const [scenarios, setScenarios] = useState<string[]>(["empty"]);
  const [policy, setPolicy] = useState("oracle");
  const [spec, setSpec] = useState("4x4");
  const [scenario, setScenario] = useState("empty");
  const [seed, setSeed] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [status, setStatus] = useState("");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [busy, setBusy] = useState(false);

  // playback loop
  useEffect(() => {
    let raf = 0, lastT = 0;
    const loop = (ts: number) => {
      const tl = tlRef.current, replay = replayRef.current, view = viewRef.current;
      const dt = lastT ? Math.min(0.1, (ts - lastT) / 1000) : 0;
      lastT = ts;
      if (replay && view && tl.length) {
        if (playingRef.current) {
          curRef.current += 30 * speedRef.current * dt;
          if (curRef.current >= tl.length - 1) {
            curRef.current = tl.length - 1; playingRef.current = false; setPlaying(false);
          }
        }
        const ci = Math.min(Math.floor(curRef.current), tl.length - 1);
        drawScene(canvasRef.current!, view, replay, tl[ci], labelsRef.current);
        drawStrip(stripRef.current!, replay, tl[ci].st.i);
        if (scrubRef.current && document.activeElement !== scrubRef.current) scrubRef.current.value = String(ci);
        if (frameLabelRef.current) frameLabelRef.current.textContent = `${ci + 1}/${tl.length}`;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  async function run(p: string, s: number, sp: string, sc: string, ev: string) {
    setBusy(true); setStatus("running episode…");
    playingRef.current = false; setPlaying(false);
    try {
      const res = await fetch("/api/episode", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ policy: p, seed: s, spec: sp, scenario: sc, env: ev }),
      });
      const d = await res.json();
      if (d.error) { setStatus("error: " + d.error); return; }
      const replay = d as Replay; replay._policy = p;
      replayRef.current = replay;
      let gi = 0;
      tlRef.current = replay.steps.flatMap((st) =>
        st.ticks.map((bricks, j) => ({ st, bricks, base: st.base_ticks?.[j], gi: gi++ })));
      curRef.current = 0;
      const cv = canvasRef.current!;
      viewRef.current = makeView(replay.length, replay.n_courses, cv.width, cv.height);
      if (scrubRef.current) { scrubRef.current.max = String(Math.max(0, tlRef.current.length - 1)); scrubRef.current.value = "0"; }
      setMetrics(replay.metrics);
      setStatus(`${replay.steps.length} placements · ${tlRef.current.length} frames · seed ${replay.seed}`);
    } catch (e) {
      setStatus("request failed: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  // (re)load policy list for the selected env, then auto-run its oracle
  useEffect(() => {
    (async () => {
      let sc = "empty";
      try {
        const d = await (await fetch(`/api/policies?env=${env}`)).json();
        if (d.policies) {
          setPolicies(d.policies); setSpecs(d.specs);
          const scs = d.scenarios ?? ["empty"];
          setScenarios(scs); sc = scs[0]; setScenario(sc);
        }
      } catch { /* ignore */ }
      setPolicy("oracle");
      const sp = env === "robot" ? "4x3" : "4x4";  // robot: in the small-wall training range
      setSpec(sp);
      run("oracle", seed, sp, sc, env);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [env]);

  const togglePlay = () => {
    const tl = tlRef.current;
    if (!tl.length) return;
    if (curRef.current >= tl.length - 1) curRef.current = 0;
    const np = !playingRef.current;
    playingRef.current = np; setPlaying(np);
  };

  const fmtPct = (x: number) => `${(x * 100).toFixed(0)}%`;

  return (
    <>
      <header>
        <h1>atrium-sim</h1>
        <span className="sub">watch a policy lay a running-bond wall · reward = live BIM audit (±3&nbsp;mm)</span>
      </header>
      <main>
        <section className="card stage">
          <canvas ref={canvasRef} width={900} height={584} />
          <div className="controls">
            <button className="play" onClick={togglePlay}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
            <input
              type="range" ref={scrubRef} min={0} max={0} defaultValue={0}
              onInput={(e) => { curRef.current = +(e.target as HTMLInputElement).value; playingRef.current = false; setPlaying(false); }}
            />
            <span className="tag" ref={frameLabelRef} style={{ color: "var(--muted)", minWidth: 96, textAlign: "right" }}>–</span>
            <label className="small">speed{" "}
              <select defaultValue="1" onChange={(e) => { speedRef.current = parseFloat(e.target.value); }}>
                <option value="0.5">0.5×</option><option value="1">1×</option>
                <option value="2">2×</option><option value="4">4×</option>
              </select>
            </label>
            <label className="small">
              <input type="checkbox" onChange={(e) => { labelsRef.current = e.target.checked; }} /> mm labels
            </label>
          </div>
          <h3>per-step reward</h3>
          <canvas ref={stripRef} className="strip" width={900} height={46} />
          <div className="legend">
            <span><span className="sw" style={{ background: "#5fb45a" }} />within ±3&nbsp;mm</span>
            <span><span className="sw" style={{ background: "#e0a030" }} />close</span>
            <span><span className="sw" style={{ background: "#d64b45" }} />out of tolerance</span>
            <span><span className="sw" style={{ background: "#5a2d28" }} />stray / toppled</span>
            <span><span className="sw" style={{ border: "1px dashed #6e737d", background: "transparent" }} />blueprint target</span>
          </div>
        </section>

        <aside className="side">
          <section className="card">
            <h3>generate replay</h3>
            <div className="metrics" style={{ gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="small">environment</label>
                <select value={env} onChange={(e) => setEnv(e.target.value as "bricklayer" | "robot")}>
                  <option value="bricklayer">wall (fixed placer)</option>
                  <option value="robot">mobile robot (reach + move)</option>
                </select>
              </div>
              <div className="field">
                <label className="small">policy</label>
                <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                  {policies.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
              <div className="row">
                <div className="field" style={{ flex: 1 }}>
                  <label className="small">wall</label>
                  <select value={spec} onChange={(e) => setSpec(e.target.value)}>
                    {specs.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div className="field" style={{ width: 96 }}>
                  <label className="small">seed</label>
                  <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
                </div>
              </div>
              <div className="field">
                <label className="small">scenario</label>
                <select value={scenario} onChange={(e) => setScenario(e.target.value)}
                        disabled={scenarios.length <= 1}>
                  {scenarios.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <button className="primary" disabled={busy}
                      onClick={() => run(policy, seed, spec, scenario, env)}>
                {busy ? "Running…" : "Generate"}
              </button>
            </div>
            <div className="status">{status}</div>
          </section>

          <section className="card">
            <h3>this run</h3>
            <div className="metrics">
              <div className="metric">
                <span className={"v " + (metrics ? (metrics.frac_in_tol >= 0.9 ? "good" : metrics.frac_in_tol < 0.3 ? "bad" : "") : "")}>
                  {metrics ? fmtPct(metrics.frac_in_tol) : "–"}
                </span><span className="k">in ±3mm</span>
              </div>
              <div className="metric"><span className="v">{metrics ? fmtPct(metrics.frac_filled) : "–"}</span><span className="k">filled</span></div>
              <div className="metric">
                <span className={"v " + (metrics ? (metrics.episode_return >= 0 ? "good" : "bad") : "")}>
                  {metrics ? (metrics.episode_return >= 0 ? "+" : "") + metrics.episode_return.toFixed(2) : "–"}
                </span><span className="k">return</span>
              </div>
              <div className="metric"><span className="v">{metrics ? metrics.waste_count.toFixed(0) : "–"}</span><span className="k">waste</span></div>
              <div className="metric"><span className="v">{metrics ? metrics.mean_abs_dev_mm.toFixed(1) + " mm" : "–"}</span><span className="k">mean dev</span></div>
              <div className="metric"><span className="v">{metrics ? metrics.placements.toFixed(0) : "–"}</span><span className="k">placements</span></div>
            </div>
          </section>
        </aside>
      </main>
    </>
  );
}
