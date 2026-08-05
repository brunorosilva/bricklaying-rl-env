"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ReplayViewer } from "@/components/ReplayViewer";
import { MetricsPanel } from "@/components/MetricsPanel";
import {
  LIVE_API_BASE, loadManifest, loadTrace, runLiveEpisode, traceKey,
  type Manifest,
} from "@/lib/traces";
import type { Metrics, Replay } from "@/lib/replay/types";

export default function ReplayPage() {
  return (
    <Suspense fallback={<main className="p-6 text-muted">loading…</main>}>
      <ReplayPageInner />
    </Suspense>
  );
}

type Env = "bricklayer" | "robot";

// used only until the manifest arrives (then overridden by its featured_policy /
// robot_specs) - "6x5"/"4x4" are always in the baked matrix, so the very first render
// never depends on a network round trip to pick something valid.
function defaultsFor(env: Env, manifest: Manifest | null, params: URLSearchParams) {
  return {
    policy: params.get("policy") || (manifest ? manifest.featured_policy[env] : ""),
    spec: params.get("spec") || (env === "robot" ? "6x5" : "4x4"),
    scenario: params.get("scenario") || "empty",
    seed: Number(params.get("seed") ?? 0),
  };
}

function ReplayPageInner() {
  const router = useRouter();
  const params = useSearchParams();

  const [env, setEnv] = useState<Env>(params.get("env") === "bricklayer" ? "bricklayer" : "robot");
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [policy, setPolicy] = useState(params.get("policy") ?? "");
  const [spec, setSpec] = useState(params.get("spec") ?? "");
  const [scenario, setScenario] = useState(params.get("scenario") ?? "empty");
  const [seed, setSeed] = useState(Number(params.get("seed") ?? 0));

  const [replay, setReplay] = useState<Replay | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  // Every selectable option comes from the baked matrix (scripts/export_traces.py), not a
  // live "/api/policies" call - there is no server behind this static export. Policies are
  // grouped for robot (featured/ablation/lineage/bake-off/baselines); bricklayer's pool is
  // just the three baselines. Specs/scenarios are whatever the matrix actually contains for
  // the current env, so a stale option never renders with a fixed value that has no match.
  const policyGroups = useMemo(() => {
    if (!manifest) return [];
    if (env === "bricklayer") {
      return [{ id: "baselines", policies: manifest.bricklayer_policies.map((p) => ({ id: p, label: p })) }];
    }
    return manifest.policy_groups;
  }, [manifest, env]);

  const specs = useMemo(() => {
    if (!manifest) return [];
    const set = new Set<string>();
    for (const t of Object.values(manifest.traces)) if (t.env === env) set.add(t.spec);
    return Array.from(set).sort((a, b) => {
      const ka = manifest.specs[a]?.kind ?? "wall";
      const kb = manifest.specs[b]?.kind ?? "wall";
      return ka === kb ? a.localeCompare(b) : ka.localeCompare(kb);
    });
  }, [manifest, env]);

  const scenarios = useMemo(() => {
    if (!manifest) return ["empty"];
    const set = new Set<string>();
    for (const t of Object.values(manifest.traces)) if (t.env === env) set.add(t.scenario);
    return set.size ? Array.from(set).sort() : ["empty"];
  }, [manifest, env]);

  function syncUrl(next: { env?: string; policy?: string; spec?: string; scenario?: string; seed?: number }) {
    const p = new URLSearchParams({
      env, policy, spec, scenario, seed: String(seed),
      ...Object.fromEntries(Object.entries(next).map(([k, v]) => [k, String(v)])),
    });
    router.replace(`/replay?${p.toString()}`, { scroll: false });
  }

  async function run(p: string, s: number, sp: string, sc: string, ev: Env, m: Manifest | null) {
    if (!p || !sp) return; // manifest hasn't resolved a default yet
    setBusy(true);
    setStatus("loading replay…");
    try {
      const key = traceKey(ev, p, sp, sc, s);
      const meta = m?.traces[key];
      let r: Replay;
      let source: string;
      if (meta) {
        r = await loadTrace(meta);
        source = `precomputed${meta.truncated ? " · ran out of steps before finishing" : ""}`;
      } else if (LIVE_API_BASE) {
        setStatus("not precomputed - asking the live backend…");
        r = await runLiveEpisode({ policy: p, seed: s, spec: sp, scenario: sc, env: ev });
        source = "live";
      } else {
        setStatus(
          "no precomputed replay for this combination on this static site - try a different " +
            "policy, wall size, or seed (0 is baked for most; a few have seeds 0-4).",
        );
        return;
      }
      r._policy = p;
      setReplay(r);
      setMetrics(r.metrics);
      setStatus(`${r.steps.length} placements · seed ${r.seed} · ${source}`);
    } catch (e) {
      setStatus("error: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  // load the manifest once on mount, resolve whatever the URL asked for (or the featured-
  // policy default) against it, and run that. Switching env later is handled directly in
  // the <select>'s onChange below, not here - re-deriving from `params` on every env change
  // would race router.replace's async URL update and could resolve against the OLD env's
  // query string.
  useEffect(() => {
    (async () => {
      const m = await loadManifest();
      setManifest(m);
      const d = defaultsFor(env, m, params);
      setPolicy(d.policy);
      setSpec(d.spec);
      setScenario(d.scenario);
      setSeed(d.seed);
      run(d.policy, d.seed, d.spec, d.scenario, env, m);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="mx-auto grid max-w-6xl grid-cols-1 gap-4 px-5 py-6 lg:grid-cols-[minmax(0,1fr)_300px]">
      <ReplayViewer replay={replay} />

      <aside className="flex flex-col gap-4">
        <section className="rounded-lg border border-line bg-panel">
          <h3 className="mx-3 mt-3 text-xs uppercase tracking-wide text-muted">generate replay</h3>
          <div className="flex flex-col gap-3 p-3">
            <Field label="environment">
              <select
                value={env}
                onChange={(e) => {
                  const v = e.target.value as Env;
                  const d = defaultsFor(v, manifest, new URLSearchParams());
                  setEnv(v);
                  setPolicy(d.policy);
                  setSpec(d.spec);
                  setScenario(d.scenario);
                  syncUrl({ env: v, policy: d.policy, spec: d.spec, scenario: d.scenario });
                  run(d.policy, seed, d.spec, d.scenario, v, manifest);
                }}
              >
                <option value="bricklayer">wall (fixed placer)</option>
                <option value="robot">mobile robot (reach + move)</option>
              </select>
            </Field>
            <Field label="policy">
              <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                {policyGroups.map((g) => (
                  <optgroup key={g.id} label={g.id}>
                    {g.policies.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </Field>
            <div className="flex gap-2">
              <div className="flex-1">
                <Field label="wall">
                  <select value={spec} onChange={(e) => setSpec(e.target.value)}>
                    {specs.map((s) => (
                      <option key={s} value={s}>
                        {manifest?.specs[s]?.label ?? s}
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
              <div className="w-24">
                <Field label="seed">
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(+e.target.value)}
                    className="w-full"
                  />
                </Field>
              </div>
            </div>
            <Field label="scenario">
              <select
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                disabled={scenarios.length <= 1}
              >
                {scenarios.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </Field>
            <button
              disabled={busy}
              onClick={() => {
                syncUrl({ policy, spec, scenario, seed });
                run(policy, seed, spec, scenario, env, manifest);
              }}
              className="rounded-md border border-accent bg-accent px-3 py-2 font-semibold text-[#1a1400] disabled:cursor-default disabled:opacity-50"
            >
              {busy ? "Running…" : "Generate"}
            </button>
          </div>
          <div className="min-h-[18px] px-3 pb-3 text-xs text-muted">{status}</div>
        </section>

        <MetricsPanel metrics={metrics} />
      </aside>
    </main>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-muted">{label}</span>
      {children}
    </label>
  );
}
