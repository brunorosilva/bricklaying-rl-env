"use client";

import { VIEW_MODES, type ViewMode } from "@/lib/replay/shared";

const LABEL: Record<ViewMode, string> = { "as-built": "as-built", inspect: "inspect", drawing: "drawing" };

/** Which color function (and, in the 3D renderer, which camera) is active - orthogonal to
 * the 3D/2D renderer choice in ReplayViewer. See shared.ts's brickColorRgb for what each
 * mode actually changes. */
export function ViewModeToggle({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  return (
    <div className="inline-flex overflow-hidden rounded-md border border-line text-xs">
      {VIEW_MODES.map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={`px-2.5 py-1 ${mode === m ? "bg-accent text-accent-ink font-medium" : "bg-panel-2 text-muted"}`}
        >
          {LABEL[m]}
        </button>
      ))}
    </div>
  );
}
