"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import { loadManifest, loadTrace, traceKey } from "@/lib/traces";
import { flattenReplay } from "@/lib/replay/useReplayPlayer";
import type { Frame, Replay } from "@/lib/replay/types";
import { SURVIVAL_DRIFT_MM, SURVIVAL_TILT_DEG } from "@/lib/replay/shared";
import { SceneCanvas } from "@/components/SceneCanvas";
import { Readout } from "@/components/Readout";

const SPEC = "house:uk_terrace";
// A clip around each strike, not the whole episode: ~2s of "still on centering" runway, ~4s
// of "does it hold" aftermath, played back at PLAY_SPEED so the moment is actually watchable
// instead of a single-frame flash.
const LOOKBACK_TICKS = 60;
const SETTLE_TICKS = 120;
const PLAY_SPEED = 0.25;
const PAUSE_MS = 1500;

type RegionResult = {
  index: number;
  style: string;
  centerX: number;
  span: number;
  windowStart: number;
  windowEnd: number;
  maxDriftMm: number;
  maxTiltDeg: number;
  survived: boolean | null; // null = no voussoirs found for this region (a data mismatch, not a real "unknown")
};

/** Derives each arch's own strike moment and post-strike drift PURELY from the replay's raw
 * per-tick brick poses - `ring_drift`/`survived` are computed server-side
 * (atrium_sim.arch.ring_drift/survived) but only ever reported as ONE aggregate fraction
 * across all arches (Metrics.arch_strike_survival), never per-arch. This is the same
 * before/after-the-strike comparison, just run client-side against data that was already
 * being shipped (see arch_regions/hard_bodies' `disappear` in shared.ts's module docs). */
function computeRegions(replay: Replay, tl: Frame[]): RegionResult[] {
  const regions = replay.arch_regions ?? [];
  const hardBodies = replay.hard_bodies ?? [];
  const lastGi = tl.length - 1;

  return regions.map((region) => {
    const centering = hardBodies.find(
      (hb) => hb.kind === "centering" && hb.verts.every(([x]) => x >= region.x0 - 1 && x <= region.x1 + 1),
    );
    const strikeGi = centering?.disappear ?? lastGi;
    const beforeGi = Math.max(0, strikeGi - 1);
    const windowStart = Math.max(0, strikeGi - LOOKBACK_TICKS);
    const windowEnd = Math.min(lastGi, strikeGi + SETTLE_TICKS);

    const before = new Map<number, { x: number; y: number; theta: number }>();
    for (const b of tl[beforeGi]?.bricks ?? []) {
      const [x, y, theta, kind, brickId] = b;
      if (kind === 2 && x >= region.x0 && x <= region.x1) before.set(brickId, { x, y, theta });
    }
    const afterById = new Map((tl[lastGi]?.bricks ?? []).map((b) => [b[4], b] as const));

    let maxDrift = 0;
    let maxTilt = 0;
    for (const [brickId, b0] of before) {
      const after = afterById.get(brickId);
      if (!after) {
        maxDrift = Infinity; // fell out of the world entirely - definitely not a survival
        maxTilt = Infinity;
        continue;
      }
      maxDrift = Math.max(maxDrift, Math.hypot(after[0] - b0.x, after[1] - b0.y));
      const dtheta = (((after[2] - b0.theta) * 180) / Math.PI + 180) % 360;
      maxTilt = Math.max(maxTilt, Math.abs(((dtheta + 360) % 360) - 180));
    }

    return {
      index: region.index,
      style: region.style,
      centerX: (region.x0 + region.x1) / 2,
      span: region.x1 - region.x0,
      windowStart,
      windowEnd,
      maxDriftMm: maxDrift,
      maxTiltDeg: maxTilt,
      survived: before.size === 0 ? null : maxDrift < SURVIVAL_DRIFT_MM && maxTilt < SURVIVAL_TILT_DEG,
    };
  });
}

/** A playhead that loops [windowStart, windowEnd] at `speed`, pausing briefly at the end
 * before looping back - independent of the other panels' playheads, since each arch strikes
 * at a different global tick and there is no single shared "now" across all three the way
 * the Compare page's synchronized diptych has. */
function useWindowedPlayhead(windowStart: number, windowEnd: number, speed: number) {
  const curRef = useRef(windowStart);
  useEffect(() => {
    let raf = 0;
    let lastT = 0;
    let pauseStart: number | null = null;
    curRef.current = windowStart;
    const tick = (ts: number) => {
      const dt = lastT ? Math.min(0.1, (ts - lastT) / 1000) : 0;
      lastT = ts;
      if (pauseStart !== null) {
        if (ts - pauseStart >= PAUSE_MS) {
          pauseStart = null;
          curRef.current = windowStart;
        }
      } else {
        curRef.current += 30 * speed * dt;
        if (curRef.current >= windowEnd) {
          curRef.current = windowEnd;
          pauseStart = ts;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [windowStart, windowEnd, speed]);
  return curRef;
}

const STYLE_LABEL: Record<string, string> = {
  semicircular: "semicircular",
  segmental: "segmental",
  jack: "jack (flat)",
};

function ArchPanel({ replay, tlRef, region }: { replay: Replay; tlRef: RefObject<Frame[]>; region: RegionResult }) {
  const curRef = useWindowedPlayhead(region.windowStart, region.windowEnd, PLAY_SPEED);
  const labelsRef = useRef(false);
  const focus = useMemo(() => ({ centerX: region.centerX, span: Math.max(region.span * 1.6, 900) }), [region.centerX, region.span]);

  const tone = region.survived === null ? undefined : region.survived ? "good" : "bad";
  const verdict = region.survived === null ? "no data" : region.survived ? "holds" : "fails";

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-ink">{STYLE_LABEL[region.style] ?? region.style}</h3>
        <span className={`text-xs font-medium uppercase tracking-wide ${tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : "text-muted"}`}>
          {verdict}
        </span>
      </div>
      <SceneCanvas
        replay={replay}
        tlRef={tlRef}
        curRef={curRef}
        labelsRef={labelsRef}
        mode="as-built"
        focus={focus}
        heightClassName="h-[34vh] min-h-[260px] w-full"
      />
      <div className="flex gap-6">
        <Readout value={Number.isFinite(region.maxDriftMm) ? region.maxDriftMm.toFixed(1) : "off"} unit="mm drift" size="sm" tone={tone} />
        <Readout value={Number.isFinite(region.maxTiltDeg) ? region.maxTiltDeg.toFixed(1) : "off"} unit="° tilt" size="sm" tone={tone} />
      </div>
    </div>
  );
}

export default function StrikePage() {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const tlRef = useRef<Frame[]>([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const manifest = await loadManifest();
      if (!manifest) {
        if (!cancelled) setError("no trace matrix available");
        return;
      }
      const policy = manifest.featured_policy.robot;
      const meta = manifest.traces[traceKey("robot", policy, SPEC, "empty", 0)];
      if (!meta) {
        if (!cancelled) setError("this case isn't baked into this build");
        return;
      }
      try {
        const r = await loadTrace(meta);
        r._policy = policy;
        if (!cancelled) {
          tlRef.current = flattenReplay(r);
          setReplay(r);
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const regions = useMemo(() => (replay ? computeRegions(replay, tlRef.current) : []), [replay]);
  const survivedCount = regions.filter((r) => r.survived).length;

  return (
    <main className="mx-auto max-w-6xl px-5 py-8">
      <div className="mb-6 max-w-2xl">
        <h1 className="text-xl font-semibold text-ink">The strike</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Once a voussoir ring closes at the keystone, the temporary timber centering under it
          is removed - the ring either stands on its own or it doesn&rsquo;t. All three
          openings on this facade close their rings; only some of them survive being struck.
        </p>
        {replay && (
          <p className="mt-3">
            <Readout value={`${survivedCount} of ${regions.length}`} unit="rings survived their strike" size="lg" tone={survivedCount === regions.length ? "good" : undefined} />
          </p>
        )}
      </div>

      {error && <p className="text-sm text-muted">{error}</p>}
      {!replay && !error && <p className="text-sm text-muted">loading a replay…</p>}

      {replay && regions.length > 0 && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {regions.map((r) => (
            <ArchPanel key={r.index} replay={replay} tlRef={tlRef} region={r} />
          ))}
        </div>
      )}

      {replay && regions.length === 0 && (
        <p className="text-sm text-muted">this replay has no structural arches to strike.</p>
      )}
    </main>
  );
}
