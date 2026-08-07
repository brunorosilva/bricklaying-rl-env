"use client";

import { useState } from "react";
import { useReplayPlayer } from "@/lib/replay/useReplayPlayer";
import type { Replay } from "@/lib/replay/types";
import type { ViewMode } from "@/lib/replay/shared";
import { StageCanvas } from "./StageCanvas";
import { SceneCanvas } from "./SceneCanvas";
import { PlaybackControls } from "./PlaybackControls";
import { RewardStrip } from "./RewardStrip";
import { Legend } from "./Legend";
import { ViewModeToggle } from "./ViewModeToggle";

/** The stage + controls + reward strip + legend, wired to one replay. Used by both the
 * /replay page (fetches by policy/spec/seed) and /build (fetches a custom plan) - the
 * viewer itself doesn't know or care where the replay came from.
 *
 * Two renderers share ONE playhead (useReplayPlayer's tlRef/curRef): the original 2D
 * canvas port of atrium_sim/render/renderer.py, and a react-three-fiber 3D scene that
 * extrudes the same poses into real solids. Only one is mounted at a time - switching
 * doesn't reset playback, since the ref-based playhead lives above both.
 *
 * `renderer` (3D/2D) and `mode` (as-built/inspect/drawing) are orthogonal choices: the
 * renderer picks WHICH engine draws the scene, the mode picks what its color function does
 * (see shared.ts's brickColorRgb) - both renderers read the same `mode` so switching the
 * renderer mid-playback never changes what you're looking at, just how it's drawn. */
export function ReplayViewer({ replay }: { replay: Replay | null }) {
  const player = useReplayPlayer(replay);
  const [renderer, setRenderer] = useState<"3d" | "2d">("3d");
  const [mode, setMode] = useState<ViewMode>("as-built");

  return (
    <section className="rounded-lg border border-line bg-panel p-2.5">
      <div className="mb-2 flex flex-wrap items-center justify-end gap-2">
        <ViewModeToggle
          mode={mode}
          onChange={(m) => {
            setMode(m);
            player.setMode(m); // keeps the 2D renderer's ref-based draw loop in sync too
          }}
        />
        <div className="inline-flex overflow-hidden rounded-md border border-line text-xs">
          <button
            onClick={() => setRenderer("3d")}
            className={`px-2.5 py-1 ${renderer === "3d" ? "bg-accent text-accent-ink font-medium" : "bg-panel-2 text-muted"}`}
          >
            3D
          </button>
          <button
            onClick={() => setRenderer("2d")}
            className={`px-2.5 py-1 ${renderer === "2d" ? "bg-accent text-accent-ink font-medium" : "bg-panel-2 text-muted"}`}
          >
            2D
          </button>
        </div>
      </div>
      {/* Both stay mounted (just hidden) rather than swapping in/out: StageCanvas's DOM node
          identity must stay stable for its ResizeObserver/view-transform effect (tied to
          [replay], not [renderer]) to keep working correctly if the user toggles back to 2D. */}
      <div className={renderer === "3d" ? "" : "hidden"}>
        <SceneCanvas
          replay={replay}
          tlRef={player.tlRef}
          curRef={player.curRef}
          labelsRef={player.labelsRef}
          mode={mode}
          active={renderer === "3d"}
        />
      </div>
      <div className={renderer === "2d" ? "" : "hidden"}>
        <StageCanvas canvasRef={player.canvasRef} />
      </div>
      <PlaybackControls
        playing={player.playing}
        frameCount={player.frameCount}
        scrubRef={player.scrubRef}
        frameLabelRef={player.frameLabelRef}
        onToggle={player.togglePlay}
        onSeek={player.seek}
        onSpeed={player.setSpeed}
        onLabels={player.setLabels}
      />
      <RewardStrip stripRef={player.stripRef} />
      <Legend mode={mode} />
    </section>
  );
}
