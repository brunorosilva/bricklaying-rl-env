"use client";

import type { RefObject } from "react";

/** The replay stage itself - a resizable canvas filling its container. Sizing/DPR/drawing
 * are owned by useReplayPlayer; this component is just the DOM element. */
export function StageCanvas({ canvasRef }: { canvasRef: RefObject<HTMLCanvasElement | null> }) {
  return (
    <canvas
      ref={canvasRef}
      className="block h-[60vh] min-h-[360px] w-full rounded-md bg-stage-bg md:h-[65vh]"
      aria-label="Bricklaying replay stage"
    />
  );
}
