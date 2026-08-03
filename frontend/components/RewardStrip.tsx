"use client";

import type { RefObject } from "react";

export function RewardStrip({ stripRef }: { stripRef: RefObject<HTMLCanvasElement | null> }) {
  return (
    <div className="px-3">
      <h3 className="mb-1 mt-3 text-xs uppercase tracking-wide text-muted">per-step reward</h3>
      <canvas ref={stripRef} className="block h-12 w-full" />
    </div>
  );
}
