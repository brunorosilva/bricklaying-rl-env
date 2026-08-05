// Client data layer for the static export: reads the baked trace matrix
// (scripts/export_traces.py -> frontend/public/traces/) instead of hitting a Next.js API
// route (there is none - `output: "export"` has no server). A live backend is optional:
// when NEXT_PUBLIC_API_BASE is set (an HF Space running webviz/api.py), runLiveEpisode()
// covers combinations the static matrix doesn't have - arbitrary seeds, and the /build
// grid editor, which can't be precomputed since a plan is drawn on the fly. If it's unset
// or the request fails, callers fall back to "not available on this static site" rather
// than a broken fetch - the site must stay fully usable with the matrix alone.
import type { Replay } from "./replay/types";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";
export const LIVE_API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "").replace(/\/+$/, "");

export type SpecMeta = { label: string; kind: "wall" | "house" };
export type PolicyEntry = { id: string; label: string };
export type PolicyGroup = { id: string; policies: PolicyEntry[] };
export type TraceMeta = {
  file: string;
  env: "robot" | "bricklayer";
  policy: string;
  spec: string;
  scenario: string;
  seed: number;
  gz_bytes: number;
  raw_bytes: number;
  steps: number;
  ticks: number;
  truncated: boolean;
  metrics: Record<string, number>;
};
export type Manifest = {
  schema: number;
  git_sha: string;
  generated_at: string;
  versions: Record<string, string>;
  featured_policy: { robot: string; bricklayer: string };
  specs: Record<string, SpecMeta>;
  robot_specs: string[];
  bricklayer_specs: string[];
  policy_groups: PolicyGroup[];
  bricklayer_policies: string[];
  traces: Record<string, TraceMeta>;
  skipped: { key: string; error: string }[];
};

export function traceKey(env: string, policy: string, spec: string, scenario: string, seed: number): string {
  return `${env}|${policy}|${spec}|${scenario}|${seed}`;
}

let manifestPromise: Promise<Manifest | null> | null = null;

/** Fetched once per page load and memoized - every page that needs it (CaseGallery,
 * /replay, /build) shares the same in-flight request. */
export function loadManifest(): Promise<Manifest | null> {
  if (!manifestPromise) {
    manifestPromise = fetch(`${BASE_PATH}/traces/index.json`)
      .then((r) => (r.ok ? (r.json() as Promise<Manifest>) : null))
      .catch(() => null);
  }
  return manifestPromise;
}

const bufferCache = new Map<string, ArrayBuffer>(); // keyed by filename, ~100KB each

async function inflate(buf: ArrayBuffer): Promise<string> {
  const bytes = new Uint8Array(buf);
  const isGzip = bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
  if (!isGzip) return new TextDecoder().decode(bytes);
  if (typeof DecompressionStream === "undefined") {
    throw new Error("this browser can't decompress gzip (no DecompressionStream) - try a recent Chrome/Firefox/Safari");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).text();
}

/** Loads and parses one precomputed replay. The compressed bytes are cached (not the
 * parsed object - a house replay can flatten into tens of MB of JS heap once
 * useReplayPlayer flattens it tick-by-tick, so an unbounded parsed cache risks OOMing a
 * mobile tab over a long browsing session). */
export async function loadTrace(meta: TraceMeta): Promise<Replay> {
  let buf = bufferCache.get(meta.file);
  if (!buf) {
    const res = await fetch(`${BASE_PATH}/traces/${meta.file}`);
    if (!res.ok) throw new Error(`trace fetch failed: HTTP ${res.status}`);
    buf = await res.arrayBuffer();
    bufferCache.set(meta.file, buf);
  }
  return JSON.parse(await inflate(buf)) as Replay;
}

export type LivePolicies = { policies: string[]; specs: string[]; scenarios: string[] };

/** How long to wait on the live backend before giving up. A cold HF Space on CPU Basic
 * needs ~30-60s to wake (container restart plus webviz/api.py's startup checkpoint
 * warm-up), and a full facade/house episode is real CPU work on top of that, plus
 * webviz/api.py serializes episodes behind a Semaphore(1) - a request can be queued behind
 * another visitor's. Past this the Space is wedged rather than slow, and hanging the button
 * forever is worse than saying so. */
const LIVE_EPISODE_TIMEOUT_MS = 120_000;
/** GET /policies is a cheap lookup against caches api.py warmed at boot - if it can't
 * answer quickly the Space is asleep, and /build's caller already falls back to the
 * oracle/random default. */
const LIVE_POLICIES_TIMEOUT_MS = 10_000;

const ASLEEP_MESSAGE = "the live backend looks like it's asleep or starting up - try again in ~30s";

export async function fetchLivePolicies(env: string): Promise<LivePolicies | null> {
  if (!LIVE_API_BASE) return null;
  try {
    const res = await fetch(`${LIVE_API_BASE}/policies?env=${encodeURIComponent(env)}`, {
      signal: AbortSignal.timeout(LIVE_POLICIES_TIMEOUT_MS),
    });
    if (!res.ok) return null;
    return (await res.json()) as LivePolicies;
  } catch {
    return null; // includes the timeout - a sleeping Space is indistinguishable from a down one
  }
}

/** Runs a fresh episode on the live backend (an HF Space, when configured) - for anything
 * the static matrix doesn't cover: an arbitrary seed, or a /build grid-editor plan. A Space
 * on CPU Basic sleeps after 48h idle, so a waking one answers with HF's own HTML page
 * instead of JSON (checked explicitly, rather than blindly calling res.json() and
 * surfacing a cryptic "Unexpected token '<'"), and a Space that isn't answering at all is
 * caught by the timeout instead of hanging the button forever. */
export async function runLiveEpisode(body: Record<string, unknown>): Promise<Replay> {
  if (!LIVE_API_BASE) throw new Error("no live backend is configured for this deployment");
  let res: Response;
  try {
    res = await fetch(`${LIVE_API_BASE}/episode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(LIVE_EPISODE_TIMEOUT_MS),
    });
  } catch (e) {
    // AbortSignal.timeout rejects with a DOMException named "TimeoutError"; anything else
    // is a network/DNS/CORS failure. Both look identical to a visitor - nothing came back -
    // and on this deployment a sleeping Space is by far the likeliest cause of either.
    if ((e as Error)?.name === "TimeoutError") throw new Error(ASLEEP_MESSAGE);
    throw new Error(`couldn't reach the live backend: ${(e as Error)?.message ?? "network error"}`);
  }
  // This API answers JSON and only JSON, so a non-JSON body is HF's own building/sleeping/
  // error page - which it serves with a 200 as readily as a 503, hence no status check here.
  if (!(res.headers.get("content-type") || "").includes("application/json")) {
    throw new Error(ASLEEP_MESSAGE);
  }
  const d = (await res.json()) as Record<string, unknown>;
  if (d.error) throw new Error(String(d.error));
  // FastAPI answers a request-validation failure with 422 + {"detail": [...]} - JSON, but
  // not a replay and not our own {"error": ...} shape. Catch it here instead of letting the
  // caller die on r.steps.length.
  if (!Array.isArray(d.steps)) {
    throw new Error(`unexpected response from the live backend: ${JSON.stringify(d).slice(0, 160)}`);
  }
  return d as unknown as Replay;
}
