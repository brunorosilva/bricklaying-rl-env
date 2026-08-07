import type { Metrics } from "@/lib/replay/types";
import { Readout } from "./Readout";

function fmtPct(x: number) {
  return `${(x * 100).toFixed(0)}%`;
}

function tone(kind: "in_tol" | "return", metrics: Metrics): "good" | "bad" | undefined {
  if (kind === "in_tol") {
    if (metrics.frac_in_tol >= 0.9) return "good";
    if (metrics.frac_in_tol < 0.3) return "bad";
    return undefined;
  }
  return metrics.episode_return >= 0 ? "good" : "bad";
}

export function MetricsPanel({ metrics }: { metrics: Metrics | null }) {
  const cells: { key: string; label: string; value: string; tone?: "good" | "bad" }[] = metrics
    ? [
        { key: "in_tol", label: "in ±3mm", value: fmtPct(metrics.frac_in_tol), tone: tone("in_tol", metrics) },
        { key: "filled", label: "filled", value: fmtPct(metrics.frac_filled) },
        { key: "return", label: "return", value: `${metrics.episode_return >= 0 ? "+" : ""}${metrics.episode_return.toFixed(2)}`, tone: tone("return", metrics) },
        { key: "waste", label: "waste", value: metrics.waste_count.toFixed(0) },
        { key: "dev", label: "mean dev", value: metrics.mean_abs_dev_mm.toFixed(1), tone: undefined },
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
            <Readout value={c.value} unit={c.key === "dev" ? "mm" : undefined} size="md" tone={c.tone} />
            <span className="text-[11px] uppercase tracking-wide text-muted">{c.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
