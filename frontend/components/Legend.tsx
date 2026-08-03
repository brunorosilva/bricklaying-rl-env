const ITEMS: { swatch: string; label: string; dashed?: boolean }[] = [
  { swatch: "#5fb45a", label: "within ±3 mm" },
  { swatch: "#e0a030", label: "close" },
  { swatch: "#d64b45", label: "out of tolerance" },
  { swatch: "#5a2d28", label: "stray / toppled" },
  { swatch: "transparent", label: "blueprint target", dashed: true },
];

export function Legend() {
  return (
    <div className="flex flex-wrap gap-3.5 px-3 pb-3 text-xs text-muted">
      {ITEMS.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{
              background: it.swatch,
              border: it.dashed ? "1px dashed #6e737d" : undefined,
            }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}
