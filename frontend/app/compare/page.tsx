"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import { loadManifest, loadTrace, traceKey, type Manifest } from "@/lib/traces";
import { flattenReplay } from "@/lib/replay/useReplayPlayer";
import type { Frame, Replay } from "@/lib/replay/types";
import { DiptychCanvas } from "@/components/DiptychCanvas";
import { Readout } from "@/components/Readout";

const SPEC = "house:uk_terrace";

/** Picks the earliest (lowest-numbered) lineage checkpoint, e.g. "robot8_v2" over
 * "robot11_v2"/"robot16_v2" - NOT policy_groups.lineage.policies[0]. That array is in
 * list_robot_checkpoints()'s alphabetic STRING order ("robot11" < "robot16" < "robot8", since
 * '1' < '8' as characters), which is the reverse of training-chronological order for exactly
 * this id scheme - array[0] would silently pick a LATER, closer-to-robot18 checkpoint instead
 * of the dramatic early-failure one this page wants. */
function pickEarliestLineagePolicy(manifest: Manifest): string | null {
  const group = manifest.policy_groups.find((g) => g.id === "lineage");
  if (!group) return null;
  let best: { id: string; gen: number } | null = null;
  for (const p of group.policies) {
    const m = /robot(\d+)/.exec(p.id);
    if (!m) continue;
    const gen = parseInt(m[1], 10);
    if (!best || gen < best.gen) best = { id: p.id, gen };
  }
  return best?.id ?? group.policies[0]?.id ?? null;
}

function policyLabel(manifest: Manifest, policyId: string): string {
  for (const g of manifest.policy_groups) {
    const found = g.policies.find((p) => p.id === policyId);
    if (found) return found.label;
  }
  return policyId;
}

/** One shared clock driving both walls, advanced by absolute tick count (not a normalized
 * 0..1 fraction) - deliberately: if the earlier policy deadlocks or gives up early, its side
 * should visibly freeze while the other keeps building, which is a truer read of "one of
 * these finishes and one doesn't" than forcing both to reach their own end at the same
 * synthetic moment. */
function useDiptychPlayhead(lenA: number, lenB: number, speed: number) {
  const curA = useRef(0);
  const curB = useRef(0);
  const elapsedRef = useRef(0);
  const playingRef = useRef(true);
  const [playing, setPlaying] = useState(true);
  playingRef.current = playing;
  const maxLen = Math.max(lenA, lenB, 1);

  useEffect(() => {
    let raf = 0;
    let lastT = 0;
    elapsedRef.current = 0;
    const tick = (ts: number) => {
      const dt = lastT ? Math.min(0.1, (ts - lastT) / 1000) : 0;
      lastT = ts;
      if (playingRef.current) {
        elapsedRef.current = Math.min(maxLen - 1, elapsedRef.current + 30 * speed * dt);
      }
      curA.current = Math.min(elapsedRef.current, Math.max(0, lenA - 1));
      curB.current = Math.min(elapsedRef.current, Math.max(0, lenB - 1));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lenA, lenB, speed]);

  const restart = () => {
    elapsedRef.current = 0;
  };
  const atEnd = () => elapsedRef.current >= maxLen - 1;

  return { curA, curB, playing, setPlaying, restart, atEnd };
}

function SideMetrics({ label, replay, curRef, otherLen }: { label: string; replay: Replay; curRef: RefObject<number>; otherLen: number }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 200);
    return () => clearInterval(id);
  }, []);
  const tl = useMemo(() => flattenReplay(replay), [replay]);
  const ci = Math.min(Math.max(0, Math.floor(curRef.current)), tl.length - 1);
  const st = tl[ci]?.st;
  // "did THIS side's episode end before the other one's" - read here (not passed down as a
  // precomputed prop) so it benefits from the SAME interval-driven re-render as the rest of
  // this component; curRef itself is a ref, not reactive state, so nothing here updates
  // without that periodic tick.
  const finished = tl.length > 0 && curRef.current >= tl.length - 1 && tl.length < otherLen;
  void tick;

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs uppercase tracking-wide text-muted">{label}</span>
      <div className="flex gap-5">
        <Readout value={st ? `${(st.frac_in_tol * 100).toFixed(0)}%` : "—"} unit="in tol" size="md" />
        <Readout value={st ? `${(st.frac_filled * 100).toFixed(0)}%` : "—"} unit="filled" size="md" />
      </div>
      {finished && <span className="text-xs text-muted">episode ended here</span>}
    </div>
  );
}

export default function ComparePage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [replayA, setReplayA] = useState<Replay | null>(null);
  const [replayB, setReplayB] = useState<Replay | null>(null);
  const [labelA, setLabelA] = useState("");
  const [labelB, setLabelB] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState(2);

  const tlRefA = useRef<Frame[]>([]);
  const tlRefB = useRef<Frame[]>([]);
  const labelsRefA = useRef(false);
  const labelsRefB = useRef(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const m = await loadManifest();
      if (!m) {
        if (!cancelled) setError("no trace matrix available");
        return;
      }
      const before = pickEarliestLineagePolicy(m);
      const after = m.featured_policy.robot;
      if (!before) {
        if (!cancelled) setError("no lineage checkpoints baked into this build");
        return;
      }
      const metaA = m.traces[traceKey("robot", before, SPEC, "empty", 0)];
      const metaB = m.traces[traceKey("robot", after, SPEC, "empty", 0)];
      if (!metaA || !metaB) {
        if (!cancelled) setError("this comparison isn't baked into this build");
        return;
      }
      try {
        const [rA, rB] = await Promise.all([loadTrace(metaA), loadTrace(metaB)]);
        rA._policy = before;
        rB._policy = after;
        if (!cancelled) {
          tlRefA.current = flattenReplay(rA);
          tlRefB.current = flattenReplay(rB);
          setReplayA(rA);
          setReplayB(rB);
          setLabelA(policyLabel(m, before));
          setLabelB(policyLabel(m, after));
          setManifest(m);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const lenA = tlRefA.current.length;
  const lenB = tlRefB.current.length;
  const { curA, curB, playing, setPlaying, restart, atEnd } = useDiptychPlayhead(lenA, lenB, speed);

  return (
    <main className="mx-auto max-w-6xl px-5 py-8">
      <div className="mb-6 max-w-2xl">
        <h1 className="text-xl font-semibold text-ink">Same wall, same seed, two policies</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          {manifest ? (
            <>
              <strong className="text-ink">{labelA}</strong> - an early lineage checkpoint - beside{" "}
              <strong className="text-ink">{labelB}</strong>, the current featured policy. Same facade, same seed,
              one shared clock: the earlier policy freezes wherever its own episode ended while the other keeps
              building.
            </>
          ) : (
            "An early lineage checkpoint beside the current featured policy, on the same facade and seed."
          )}
        </p>
      </div>

      {error && <p className="text-sm text-muted">{error}</p>}
      {!replayA && !replayB && !error && <p className="text-sm text-muted">loading two replays…</p>}

      {replayA && replayB && (
        <>
          <DiptychCanvas
            replayA={replayA} tlRefA={tlRefA} curRefA={curA} labelsRefA={labelsRefA}
            replayB={replayB} tlRefB={tlRefB} curRefB={curB} labelsRefB={labelsRefB}
          />
          <div className="mt-3 flex flex-wrap items-center gap-4 border-t border-line pt-3">
            <button
              onClick={() => {
                if (!playing && atEnd()) restart();
                setPlaying((p) => !p);
              }}
              className="min-w-[84px] rounded-md border border-line bg-panel-2 px-3 py-1.5 font-medium text-ink hover:border-muted"
            >
              {playing ? "❚❚ Pause" : "▶ Replay"}
            </button>
            <label className="flex items-center gap-1.5 text-xs text-muted">
              speed
              <select
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="rounded-md border border-line bg-panel-2 px-2 py-1 text-ink"
              >
                <option value="0.5">0.5×</option>
                <option value="1">1×</option>
                <option value="2">2×</option>
              </select>
            </label>
          </div>
          <div className="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2">
            <SideMetrics label={labelA} replay={replayA} curRef={curA} otherLen={lenB} />
            <SideMetrics label={labelB} replay={replayB} curRef={curB} otherLen={lenA} />
          </div>
        </>
      )}
    </main>
  );
}
