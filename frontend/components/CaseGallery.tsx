import Link from "next/link";

type Case = {
  title: string;
  description: string;
  env: "bricklayer" | "robot";
  spec: string;
  tag?: string;
};

const CASES: Case[] = [
  {
    title: "Small wall",
    description: "A 4×4 wall, fully within the arm's reach - the baseline placement task.",
    env: "bricklayer",
    spec: "4x4",
  },
  {
    title: "Learning to move",
    description: "8×5 - wider than the robot's ~500mm reach, so it has to walk to finish.",
    env: "robot",
    spec: "8x5",
    tag: "mobile robot",
  },
  {
    title: "Zero-shot generalization",
    description: "20×10 (205 bricks) - far beyond any curriculum rung the policy trained on.",
    env: "robot",
    spec: "20x10",
    tag: "mobile robot",
  },
  {
    title: "UK terrace",
    description: "A semicircular window, a segmental door, and a jack-arched window - three real structural arches on one facade.",
    env: "robot",
    spec: "house:uk_terrace",
    tag: "structural arches",
  },
  {
    title: "Colonial facade",
    description: "A VLM-perceived elevation from a photograph, tiled into 13 buildable panels around five openings.",
    env: "robot",
    spec: "house:colonial",
    tag: "image → plan",
  },
  {
    title: "Surprise me",
    description: "A random flat wall size, freshly seeded.",
    env: "robot",
    spec: "random",
  },
];

export function CaseGallery() {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {CASES.map((c) => (
        <Link
          key={c.title}
          href={`/replay?env=${c.env}&spec=${encodeURIComponent(c.spec)}&policy=oracle`}
          className="group flex flex-col gap-2 rounded-lg border border-line bg-panel p-4 transition-colors hover:border-accent"
        >
          <div className="flex items-center justify-between gap-2">
            <h3 className="font-medium text-ink group-hover:text-accent">{c.title}</h3>
            {c.tag && (
              <span className="shrink-0 rounded-full border border-line px-2 py-0.5 text-[10px] uppercase tracking-wide text-muted">
                {c.tag}
              </span>
            )}
          </div>
          <p className="text-sm leading-snug text-muted">{c.description}</p>
          <span className="mt-1 text-xs text-muted group-hover:text-accent">watch replay →</span>
        </Link>
      ))}
    </div>
  );
}
