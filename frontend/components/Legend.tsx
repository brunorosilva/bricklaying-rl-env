import { MATERIAL, MEASURE_HEX, PALETTE, type ViewMode } from "@/lib/replay/shared";

type Item = { swatch: string; label: string; dashed?: boolean };

const BLUEPRINT_ITEM: Item = { swatch: "transparent", label: "blueprint target", dashed: true };

const ITEMS: Record<ViewMode, Item[]> = {
  // deviation isn't shown at all here - the legend reflects that honestly rather than
  // listing a "within tolerance" swatch that as-built never actually paints.
  "as-built": [
    { swatch: MATERIAL.clay, label: "clay brick" },
    { swatch: MATERIAL.clayFallen, label: "stray / toppled" },
    BLUEPRINT_ITEM,
  ],
  inspect: [
    { swatch: MEASURE_HEX.neutral, label: "within ±3 mm" },
    { swatch: MEASURE_HEX.tealStrong, label: "under target" },
    { swatch: MEASURE_HEX.redStrong, label: "over target" },
    { swatch: MATERIAL.clayFallen, label: "stray / toppled" },
    BLUEPRINT_ITEM,
  ],
  drawing: [
    { swatch: PALETTE.chalk, label: "built" },
    BLUEPRINT_ITEM,
  ],
};

export function Legend({ mode }: { mode: ViewMode }) {
  return (
    <div className="flex flex-wrap gap-3.5 px-3 pb-3 text-xs text-muted">
      {ITEMS[mode].map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{
              background: it.swatch,
              border: it.dashed ? `1px dashed ${PALETTE.chalk}` : undefined,
            }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}
