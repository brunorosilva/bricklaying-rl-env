"use client";

import type { RefObject } from "react";

type Props = {
  playing: boolean;
  frameCount: number;
  scrubRef: RefObject<HTMLInputElement | null>;
  frameLabelRef: RefObject<HTMLSpanElement | null>;
  onToggle: () => void;
  onSeek: (frame: number) => void;
  onSpeed: (v: number) => void;
  onLabels: (v: boolean) => void;
};

export function PlaybackControls({
  playing, frameCount, scrubRef, frameLabelRef, onToggle, onSeek, onSpeed, onLabels,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-3 border-t border-line px-3 py-2.5">
      <button
        onClick={onToggle}
        className="min-w-[84px] rounded-md border border-line bg-panel-2 px-3 py-1.5 font-medium text-ink hover:border-muted disabled:cursor-default disabled:opacity-50"
      >
        {playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <input
        ref={scrubRef}
        type="range"
        min={0}
        max={Math.max(0, frameCount - 1)}
        defaultValue={0}
        onInput={(e) => onSeek(+(e.target as HTMLInputElement).value)}
        className="min-w-[120px] flex-1"
      />
      <span
        ref={frameLabelRef}
        className="min-w-24 text-right font-mono text-sm tabular-nums text-muted"
      >
        –
      </span>
      <label className="flex items-center gap-1.5 text-xs text-muted">
        speed
        <select
          defaultValue="1"
          onChange={(e) => onSpeed(parseFloat(e.target.value))}
          className="rounded-md border border-line bg-panel-2 px-2 py-1 text-ink"
        >
          <option value="0.5">0.5×</option>
          <option value="1">1×</option>
          <option value="2">2×</option>
          <option value="4">4×</option>
        </select>
      </label>
      <label className="flex items-center gap-1.5 text-xs text-muted">
        <input type="checkbox" onChange={(e) => onLabels(e.target.checked)} />
        mm labels
      </label>
    </div>
  );
}
