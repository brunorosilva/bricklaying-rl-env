import type { Metrics } from "@/lib/replay/types";

function fmtPct(x: number) {
  return `${(x * 100).toFixed(0)}%`;
}

function toneClass(kind: "in_tol" | "return", metrics: Metrics): string {
  if (kind === "in_tol") {
    if (metrics.frac_in_tol >= 0.9) return "text-good";
    if (metrics.frac_in_tol < 0.3) return "text-bad";
    return "";
  }
  return metrics.episode_return >= 0 ? "text-good" : "text-bad";
}

export function MetricsPanel({ metrics }: { metrics: Metrics | null }) {
  const cells: { key: string; label: string; value: string; tone?: string }[] = metrics
    ? [
        { key: "in_tol", label: "in ±3mm", value: fmtPct(metrics.frac_in_tol), tone: toneClass("in_tol", metrics) },
        { key: "filled", label: "filled", value: fmtPct(metrics.frac_filled) },
        { key: "return", label: "return", value: `${metrics.episode_return >= 0 ? "+" : ""}${metrics.episode_return.toFixed(2)}`, tone: toneClass("return", metrics) },
        { key: "waste", label: "waste", value: metrics.waste_count.toFixed(0) },
        { key: "dev", label: "mean dev", value: `${metrics.mean_abs_dev_mm.toFixed(1)} mm` },
        { key: "placements", label: "placements", value: metrics.placements.toFixed(0) },
      ]
    : [];

  return (
    <section className="rounded-lg border border-line bg-panel">
      <h3 className="mx-3 mt-3 text-xs uppercase tracking-wide text-muted">this run</h3>
      <div className="grid grid-cols-2 gap-x-3 gap-y-2 p-3">
        {cells.length === 0 && <span className="col-span-2 text-sm text-muted">–</span>}
        {cells.map((c) => (
          <div key={c.key} className="flex flex-col">
            <span className={`text-xl tabular-nums ${c.tone ?? ""}`}>{c.value}</span>
            <span className="text-[11px] uppercase tracking-wide text-muted">{c.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
