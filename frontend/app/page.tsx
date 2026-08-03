import { CaseGallery } from "@/components/CaseGallery";

export default function Home() {
  return (
    <main className="mx-auto max-w-6xl px-5 py-10">
      <div className="mb-8 max-w-2xl">
        <h1 className="text-2xl font-semibold text-ink">Watch a policy lay a wall</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          A physics-based bricklaying RL environment: a mobile gantry robot places bricks
          against a blueprint and is scored like a real mason - every brick within BIM
          ±3&nbsp;mm tolerance, minimal waste, live physics, so a sloppy placement can topple
          the wall. Pick a case below, or{" "}
          <a href="/build" className="text-accent hover:underline">
            paint your own grid
          </a>
          .
        </p>
      </div>
      <CaseGallery />
    </main>
  );
}
