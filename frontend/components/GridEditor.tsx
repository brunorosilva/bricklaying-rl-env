"use client";

import { useMemo, useState } from "react";
import { PALETTE } from "@/lib/replay/shared";

export type OpeningDraft = {
  id: string;
  kind: string;
  col: number;
  row: number;
  n_cols: number;
  n_rows: number;
  has_lintel: boolean;
  has_sill: boolean;
  arch_style: "flat" | "lintel_soldier" | "semicircular" | "segmental" | "jack";
  arch_ring_courses: number;
};

// A qualitative palette for distinguishing opening regions in the editor - a different
// visual domain than the wall's own material/measurement language (closer to calendar-event
// colors than to brick color), so this draws from PALETTE's existing hues rather than the
// render system's own rules.
const SWATCHES = [PALETTE.accent, PALETTE.robot, PALETTE.clay, "#6FA9AC", "#C24A3F", PALETTE.chalk];

function newOpening(id: string, col: number, row: number): OpeningDraft {
  return {
    id, kind: "window", col, row, n_cols: 2, n_rows: 2,
    has_lintel: true, has_sill: false, arch_style: "flat", arch_ring_courses: 2,
  };
}

type Props = {
  gridCols: number;
  gridRows: number;
  onGridChange: (cols: number, rows: number) => void;
  openings: OpeningDraft[];
  onOpeningsChange: (openings: OpeningDraft[]) => void;
};

export function GridEditor({ gridCols, gridRows, onGridChange, openings, onOpeningsChange }: Props) {
  const [nextId, setNextId] = useState(0);

  // which opening (if any) covers a given cell
  const cellOwner = useMemo(() => {
    const map = new Map<string, number>(); // "col,row" -> opening index
    openings.forEach((o, idx) => {
      for (let c = o.col; c < o.col + o.n_cols; c++) {
        for (let r = o.row; r < o.row + o.n_rows; r++) map.set(`${c},${r}`, idx);
      }
    });
    return map;
  }, [openings]);

  function handleCellClick(col: number, row: number) {
    const key = `${col},${row}`;
    const owner = cellOwner.get(key);
    if (owner != null) {
      onOpeningsChange(openings.filter((_, i) => i !== owner));
      return;
    }
    const id = `o${nextId}`;
    setNextId((n) => n + 1);
    onOpeningsChange([...openings, newOpening(id, col, row)]);
  }

  function updateOpening(id: string, patch: Partial<OpeningDraft>) {
    onOpeningsChange(openings.map((o) => (o.id === id ? { ...o, ...patch } : o)));
  }

  const rows = Array.from({ length: gridRows }, (_, i) => gridRows - 1 - i); // row 0 (ground) at the bottom

  return (
    <div className="flex flex-col gap-4 lg:flex-row">
      <div className="flex-1">
        <div className="mb-3 flex gap-3">
          <NumberField label="width (modules)" value={gridCols} min={2} max={40} onChange={(v) => onGridChange(v, gridRows)} />
          <NumberField label="height (courses)" value={gridRows} min={2} max={40} onChange={(v) => onGridChange(gridCols, v)} />
        </div>
        <div
          className="grid w-full gap-px overflow-hidden rounded-md border border-line bg-line"
          style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}
        >
          {rows.map((row) =>
            Array.from({ length: gridCols }, (_, col) => {
              const owner = cellOwner.get(`${col},${row}`);
              const color = owner != null ? SWATCHES[owner % SWATCHES.length] : undefined;
              return (
                <button
                  key={`${col},${row}`}
                  onClick={() => handleCellClick(col, row)}
                  title={`col ${col}, row ${row}`}
                  className="aspect-[11/3] w-full transition-opacity hover:opacity-80"
                  style={{ background: color ?? `${PALETTE.clay}33` }}
                />
              );
            }),
          )}
        </div>
        <p className="mt-2 text-xs text-muted">
          click an empty cell to start a 2×2 opening there; click any cell of an opening to remove it.
          Row 0 is the ground course.
        </p>
      </div>

      <div className="flex w-full flex-col gap-2 lg:w-72">
        <h3 className="text-xs uppercase tracking-wide text-muted">openings ({openings.length})</h3>
        {openings.length === 0 && <p className="text-sm text-muted">none yet - click the grid to add one.</p>}
        {openings.map((o, idx) => (
          <div key={o.id} className="rounded-md border border-line bg-panel-2 p-2.5">
            <div className="mb-2 flex items-center justify-between">
              <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: SWATCHES[idx % SWATCHES.length] }} />
                opening {idx + 1}
              </span>
              <button
                onClick={() => onOpeningsChange(openings.filter((x) => x.id !== o.id))}
                className="text-xs text-bad hover:underline"
              >
                remove
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="col" value={o.col} min={0} onChange={(v) => updateOpening(o.id, { col: v })} compact />
              <NumberField label="row" value={o.row} min={0} onChange={(v) => updateOpening(o.id, { row: v })} compact />
              <NumberField label="width" value={o.n_cols} min={1} onChange={(v) => updateOpening(o.id, { n_cols: v })} compact />
              <NumberField label="height" value={o.n_rows} min={1} onChange={(v) => updateOpening(o.id, { n_rows: v })} compact />
            </div>
            <label className="mt-2 flex flex-col gap-1 text-xs text-muted">
              arch style
              <select
                value={o.arch_style}
                onChange={(e) => updateOpening(o.id, { arch_style: e.target.value as OpeningDraft["arch_style"] })}
              >
                <option value="flat">flat (plain lintel)</option>
                <option value="lintel_soldier">lintel + cosmetic soldier</option>
                <option value="semicircular">semicircular arch</option>
                <option value="segmental">segmental arch</option>
                <option value="jack">jack arch</option>
              </select>
            </label>
            {o.arch_style !== "flat" && o.arch_style !== "lintel_soldier" && (
              <NumberField
                label="ring courses"
                value={o.arch_ring_courses}
                min={1}
                max={6}
                onChange={(v) => updateOpening(o.id, { arch_ring_courses: v })}
                compact
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function NumberField({
  label, value, min, max, onChange, compact,
}: { label: string; value: number; min?: number; max?: number; onChange: (v: number) => void; compact?: boolean }) {
  return (
    <label className={`flex flex-col gap-1 text-xs text-muted ${compact ? "" : "w-32"}`}>
      {label}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Math.max(min ?? 0, Math.min(max ?? 9999, +e.target.value || 0)))}
      />
    </label>
  );
}
