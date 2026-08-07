"use client";

import { useEffect, useRef, useState } from "react";
import { GridEditor, type OpeningDraft } from "@/components/GridEditor";
import { ReplayViewer } from "@/components/ReplayViewer";
import { MetricsPanel } from "@/components/MetricsPanel";
import { LiveState, type LivePhase } from "@/components/LiveState";
import { LIVE_API_BASE, fetchLivePolicies, runLiveEpisode } from "@/lib/traces";
import type { Metrics, Replay } from "@/lib/replay/types";

const STRUCTURAL_ARCH_STYLES = new Set(["semicircular", "segmental", "jack"]);
// After this long without an answer, a request is more likely a cold Space waking up than a
// small plan still computing - see traces.ts's own LIVE_EPISODE_TIMEOUT_MS docs on why a
// waking Space is otherwise indistinguishable from a slow one until it either answers or
// actually times out.
const WAKING_THRESHOLD_MS = 3500;

export default function BuildPage() {
  const [gridCols, setGridCols] = useState(10);
  const [gridRows, setGridRows] = useState(8);
  const [openings, setOpenings] = useState<OpeningDraft[]>([]);
  const [policies, setPolicies] = useState<string[]>(["oracle", "random"]);
  const [policy, setPolicy] = useState("oracle");
  const [seed, setSeed] = useState(0);

  const [replay, setReplay] = useState<Replay | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [resultLine, setResultLine] = useState("");
  const [phase, setPhase] = useState<LivePhase | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const wakingTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const busy = phase === "sending" || phase === "waking";

  // Custom plans are drawn live, one keystroke at a time - there's no way to precompute a
  // matrix for them the way scripts/export_traces.py does for /replay's fixed cases, so this
  // page only works when a live backend (webviz/api.py on an HF Space) is configured. If it
  // isn't, everything below stays visible but disabled - see the notice under the header.
  useEffect(() => {
    if (!LIVE_API_BASE) return;
    (async () => {
      const d = await fetchLivePolicies("robot");
      if (d?.policies) setPolicies(d.policies);
    })();
  }, []);

  async function run() {
    setPhase("sending");
    setResultLine("");
    clearTimeout(wakingTimer.current);
    wakingTimer.current = setTimeout(() => setPhase("waking"), WAKING_THRESHOLD_MS);
    try {
      const plan = {
        grid_cols: gridCols,
        grid_rows: gridRows,
        openings: openings.map((o) => ({
          kind: o.kind,
          col: o.col,
          row: o.row,
          n_cols: o.n_cols,
          n_rows: o.n_rows,
          // a real structural arch IS its own head - a lintel only applies to the two
          // non-structural styles (see atrium_sim.facade.Opening's own docstring)
          has_lintel: !STRUCTURAL_ARCH_STYLES.has(o.arch_style),
          has_sill: o.has_sill,
          arch_style: o.arch_style,
          arch_ring_courses: o.arch_ring_courses,
        })),
      };
      // FacadePlan.validate() (server-side) rejects overlapping/out-of-grid openings with a
      // clear message - runLiveEpisode surfaces it inline rather than a silent failure.
      const r = await runLiveEpisode({ policy, seed, env: "robot", plan });
      r._policy = policy;
      setReplay(r);
      setMetrics(r.metrics);
      setResultLine(`${r.steps.length} placements · seed ${r.seed}`);
      setPhase(null);
    } catch (e) {
      setErrorMessage((e as Error).message);
      setPhase("error");
    } finally {
      clearTimeout(wakingTimer.current);
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-5 py-6">
      <div className="mb-4 max-w-2xl">
        <h1 className="text-xl font-semibold text-ink">Build your own facade</h1>
        <p className="mt-1 text-sm text-muted">
          Set the grid, add openings, pick an arch style for each. The remaining brickwork is
          tiled into buildable panels automatically - there is no pre-flight buildability
          check, so an unusual combination of openings may turn out to be a hard (or
          physically impossible) level. The replay is the feedback, same as any other case.
        </p>
        {!LIVE_API_BASE && (
          <div className="mt-3">
            <LiveState phase="unavailable" />
          </div>
        )}
      </div>

      <section className={`mb-4 rounded-lg border border-line bg-panel p-4 ${!LIVE_API_BASE ? "opacity-50" : ""}`}>
        <fieldset disabled={!LIVE_API_BASE}>
          <GridEditor
            gridCols={gridCols}
            gridRows={gridRows}
            onGridChange={(c, r) => {
              setGridCols(c);
              setGridRows(r);
            }}
            openings={openings}
            onOpeningsChange={setOpenings}
          />
        </fieldset>
      </section>

      <section className={`mb-6 flex flex-wrap items-end gap-3 rounded-lg border border-line bg-panel p-4 ${!LIVE_API_BASE ? "opacity-50" : ""}`}>
        <fieldset disabled={!LIVE_API_BASE} className="contents">
          <label className="flex flex-col gap-1 text-xs text-muted">
            policy
            <select value={policy} onChange={(e) => setPolicy(e.target.value)} className="w-48">
              {policies.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-xs text-muted">
            seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(+e.target.value)}
              className="w-24"
            />
          </label>
          <button
            onClick={run}
            disabled={busy}
            className="rounded-md border border-accent bg-accent px-4 py-2 font-semibold text-accent-ink disabled:cursor-default disabled:opacity-50"
          >
            {busy ? "Running…" : "Run"}
          </button>
          {!phase && resultLine && <span className="text-xs text-muted">{resultLine}</span>}
        </fieldset>
      </section>

      {phase && (
        <div className="mb-6 -mt-2">
          <LiveState phase={phase} message={phase === "error" ? errorMessage : undefined} />
        </div>
      )}

      {replay && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
          <ReplayViewer replay={replay} />
          <MetricsPanel metrics={metrics} />
        </div>
      )}
    </main>
  );
}
