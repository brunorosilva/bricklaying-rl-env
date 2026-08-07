import Link from "next/link";
import { CaseGallery } from "@/components/CaseGallery";
import { HeroStage } from "@/components/HeroStage";

export default function Home() {
  return (
    <main>
      <HeroStage />
      <div className="mx-auto max-w-6xl px-5 py-10">
        <div className="mb-8 max-w-2xl">
          <h2 className="text-xl font-semibold text-ink">Pick a case</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            Every card below is a precomputed replay - a different wall size, facade, or
            baseline policy, scored the same way: every brick within BIM ±3&nbsp;mm tolerance,
            minimal waste, live physics underneath. Or{" "}
            <Link href="/build" className="text-accent hover:underline">
              paint your own grid
            </Link>
            .
          </p>
        </div>
        <CaseGallery />
      </div>
    </main>
  );
}
