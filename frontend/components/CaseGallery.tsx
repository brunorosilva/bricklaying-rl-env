"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

type Case = {
  title: string;
  description: string;
  env: "bricklayer" | "robot";
  spec: string;
  tag?: string;
  /** if set, the card is a button that picks a random baked seed in [0, randomSeeds) on
   * click rather than a fixed href - see scripts/export_traces.py's `random` extras for
   * robot18, which bakes several seeds specifically so this card is different every time
   * without needing a live backend. */
  randomSeeds?: number;
};

// No `policy` here - every card links to /replay without one, so the page resolves it
// against the manifest's featured_policy (robot18 for the robot env) at load time. That
// keeps this list from hardcoding a timestamped checkpoint dir name that will churn as
// training continues, and keeps every card pointed at the current best policy for free.
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
    description: "17×8 (140 bricks) - far beyond any curriculum rung the policy trained on.",
    env: "robot",
    spec: "17x8",
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
    randomSeeds: 5,
  },
];

function CardBody({ c }: { c: Case }) {
  return (
    <>
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
    </>
  );
}

export function CaseGallery() {
  const router = useRouter();
  const cardClass =
    "group flex flex-col gap-2 rounded-lg border border-line bg-panel p-4 text-left transition-colors hover:border-accent";

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {CASES.map((c) => {
        const href = `/replay?env=${c.env}&spec=${encodeURIComponent(c.spec)}`;
        if (c.randomSeeds) {
          return (
            <button
              key={c.title}
              onClick={() => router.push(`${href}&seed=${Math.floor(Math.random() * c.randomSeeds!)}`)}
              className={cardClass}
            >
              <CardBody c={c} />
            </button>
          );
        }
        return (
          <Link key={c.title} href={href} className={cardClass}>
            <CardBody c={c} />
          </Link>
        );
      })}
    </div>
  );
}
