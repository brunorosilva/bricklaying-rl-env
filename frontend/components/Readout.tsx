/** The one typographic device used everywhere a number is the point: the hero, the 3D HUD,
 * MetricsPanel, the strike page's drift readout. Mono + tabular-nums + tight tracking on a
 * big value, with its unit in --muted at a noticeably smaller size - this single pattern
 * carries most of the "technical instrument" feel the rest of the identity is going for. */

const SIZE = {
  sm: { value: "text-lg", unit: "text-[11px]" },
  md: { value: "text-2xl", unit: "text-xs" },
  lg: { value: "text-4xl", unit: "text-sm" },
  xl: { value: "text-6xl", unit: "text-base" },
} as const;

type Props = {
  value: string;
  unit?: string;
  size?: keyof typeof SIZE;
  tone?: "good" | "bad" | "accent";
  className?: string;
};

export function Readout({ value, unit, size = "md", tone, className }: Props) {
  const toneClass = tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : tone === "accent" ? "text-accent" : "text-ink";
  const s = SIZE[size];
  return (
    <span className={`font-mono tabular-nums tracking-tight ${s.value} ${toneClass} ${className ?? ""}`}>
      {value}
      {unit && <span className={`ml-1.5 font-sans font-normal tracking-normal text-muted ${s.unit}`}>{unit}</span>}
    </span>
  );
}
