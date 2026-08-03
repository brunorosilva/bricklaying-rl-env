"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ReplayViewer } from "@/components/ReplayViewer";
import { MetricsPanel } from "@/components/MetricsPanel";
import type { Metrics, Replay } from "@/lib/replay/types";

export default function ReplayPage() {
  return (
    <Suspense fallback={<main className="p-6 text-muted">loading…</main>}>
      <ReplayPageInner />
    </Suspense>
  );
}

function ReplayPageInner() {
  const router = useRouter();
  const params = useSearchParams();

  const [env, setEnv] = useState<"bricklayer" | "robot">(
    params.get("env") === "bricklayer" ? "bricklayer" : "robot",
  );
  const [policies, setPolicies] = useState<string[]>([]);
  const [specs, setSpecs] = useState<string[]>([]);
  const [scenarios, setScenarios] = useState<string[]>(["empty"]);
  const [policy, setPolicy] = useState(params.get("policy") ?? "oracle");
  const [spec, setSpec] = useState(params.get("spec") ?? (env === "robot" ? "4x3" : "4x4"));
  const [scenario, setScenario] = useState(params.get("scenario") ?? "empty");
  const [seed, setSeed] = useState(Number(params.get("seed") ?? 0));

  const [replay, setReplay] = useState<Replay | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  function syncUrl(next: { env?: string; policy?: string; spec?: string; scenario?: string; seed?: number }) {
    const p = new URLSearchParams({
      env, policy, spec, scenario, seed: String(seed),
      ...Object.fromEntries(Object.entries(next).map(([k, v]) => [k, String(v)])),
    });
    router.replace(`/replay?${p.toString()}`, { scroll: false });
  }

  async function run(p: string, s: number, sp: string, sc: string, ev: string) {
    setBusy(true);
    setStatus("running episode…");
    try {
      const res = await fetch("/api/episode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ policy: p, seed: s, spec: sp, scenario: sc, env: ev }),
      });
      const d = await res.json();
      if (d.error) {
        setStatus("error: " + d.error);
        return;
      }
      const r = d as Replay;
      r._policy = p;
      setReplay(r);
      setMetrics(r.metrics);
      setStatus(`${r.steps.length} placements · seed ${r.seed}`);
    } catch (e) {
      setStatus("request failed: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  // (re)load the policy/spec/scenario list for the selected env, then run whatever the URL asked for
  useEffect(() => {
    (async () => {
      try {
        const d = await (await fetch(`/api/policies?env=${env}`)).json();
        if (d.policies) {
          setPolicies(d.policies);
          setSpecs(d.specs);
          setScenarios(d.scenarios ?? ["empty"]);
        }
      } catch {
        /* ignore */
      }
      run(policy, seed, spec, scenario, env);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [env]);

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
                  const v = e.target.value as "bricklayer" | "robot";
                  setEnv(v);
                  syncUrl({ env: v });
                }}
              >
                <option value="bricklayer">wall (fixed placer)</option>
                <option value="robot">mobile robot (reach + move)</option>
              </select>
            </Field>
            <Field label="policy">
              <select value={policy} onChange={(e) => setPolicy(e.target.value)}>
                {policies.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </Field>
            <div className="flex gap-2">
              <div className="flex-1">
                <Field label="wall">
                  <select value={spec} onChange={(e) => setSpec(e.target.value)}>
                    {specs.map((s) => (
                      <option key={s} value={s}>
                        {s}
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
                run(policy, seed, spec, scenario, env);
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
