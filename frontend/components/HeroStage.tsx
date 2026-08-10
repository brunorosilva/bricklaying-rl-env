"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useReplayPlayer } from "@/lib/replay/useReplayPlayer";
import { loadManifest, loadTrace, traceKey } from "@/lib/traces";
import type { Replay } from "@/lib/replay/types";
import { TOL_MM } from "@/lib/replay/shared";
import { SceneCanvas } from "./SceneCanvas";
import { Readout } from "./Readout";

const HERO_SPEC = "house:uk_terrace_classic";

/** The home page's full-bleed, autoplaying, looping build - the first thing a visitor sees,
 * replacing what used to be a heading and a card grid. Always the current featured robot
 * policy on the UK terrace facade (three real structural arches, the most visually rich
 * baked case) - fetched from the same static trace matrix every other page reads, so this
 * costs nothing extra to bake and never touches the live API (see traces.ts's own docs on
 * why every hero/strike/compare moment has to work this way: a cold HF Space is a ~30s wait,
 * and a visitor arriving from a shared link is exactly who'd eat it).
 *
 * Deliberately the jack-free "classic" variant, not plain `uk_terrace`: that one's jack ring
 * is a real, documented 0%-survival gap (see README) - honest as a research finding, but a
 * bad first impression as the very first thing a visitor sees autoplaying on loop. The
 * jack-arch story still gets its due on /strike and /compare, which link to it explicitly. */
export function HeroStage() {
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const manifest = await loadManifest();
      if (!manifest) {
        if (!cancelled) setError("no trace matrix available");
        return;
      }
      const policy = manifest.featured_policy.robot;
      const key = traceKey("robot", policy, HERO_SPEC, "empty", 0);
      const meta = manifest.traces[key];
      if (!meta) {
        if (!cancelled) setError("hero case not baked into this build");
        return;
      }
      try {
        const r = await loadTrace(meta);
        r._policy = policy;
        if (!cancelled) setReplay(r);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const player = useReplayPlayer(replay, { loop: true, autoplay: true });
  const metrics = replay?.metrics;

  return (
    <div className="relative h-[72vh] min-h-[440px] w-full overflow-hidden md:h-[82vh]">
      {replay ? (
        <SceneCanvas
          replay={replay}
          tlRef={player.tlRef}
          curRef={player.curRef}
          labelsRef={player.labelsRef}
          mode="as-built"
          autoRotate
          heightClassName="h-full w-full"
        />
      ) : (
        <div className="absolute inset-0 animate-pulse bg-panel" />
      )}

      {/* readable over the render either way, whether it's still loading or fully live */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-bg/90 via-transparent to-bg/30" />

      <div className="absolute inset-0 z-10 flex flex-col justify-end gap-4 px-5 pb-10 md:px-10 md:pb-14">
        <span className="text-[15px] font-semibold tracking-wide text-accent">Bricklaying with RL</span>
        <h1 className="max-w-xl text-2xl font-semibold text-ink md:text-3xl">
          A physics-based bricklaying robot, built to BIM tolerance
        </h1>
        <p className="max-w-lg text-sm leading-relaxed text-muted">
          A mobile gantry robot laying a real UK-terrace facade - three structural arches,
          judged the same way a site inspection would: every brick against the blueprint,
          live rigid-body physics underneath, so a careless placement can topple the wall.
        </p>
        <div className="flex flex-wrap items-center gap-6 pt-1">
          {metrics ? (
            <Readout
              value={`${(metrics.frac_in_tol * 100).toFixed(0)}%`}
              unit={`within ±${TOL_MM.toFixed(1)} mm`}
              size="xl"
              tone="accent"
            />
          ) : error ? (
            <span className="text-sm text-muted">{error}</span>
          ) : (
            <span className="text-sm text-muted">loading a replay…</span>
          )}
          <nav className="flex flex-wrap gap-4 text-sm">
            <Link href="/replay" className="text-ink underline decoration-line underline-offset-4 hover:decoration-accent">
              watch a replay
            </Link>
            <Link href="/strike" className="text-ink underline decoration-line underline-offset-4 hover:decoration-accent">
              the strike
            </Link>
            <Link href="/compare" className="text-ink underline decoration-line underline-offset-4 hover:decoration-accent">
              compare policies
            </Link>
          </nav>
        </div>
      </div>
    </div>
  );
}
