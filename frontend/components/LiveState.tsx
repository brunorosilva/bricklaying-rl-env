"use client";

/** Real designed states for the live backend (an HF Space, when configured) - not bare
 * strings dropped into a <span>. traces.ts already has the plumbing (timeouts, an explicit
 * HTML-instead-of-JSON check, ASLEEP_MESSAGE) for exactly these states; this is the visual
 * treatment that was missing, and the difference between a first-time visitor reading
 * "/build is broken" vs. "/build is computing". */
export type LivePhase = "sending" | "waking" | "error" | "unavailable";

const PulseDot = () => (
  <span className="relative flex h-2 w-2 shrink-0">
    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-75" />
    <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
  </span>
);

export function LiveState({ phase, message }: { phase: LivePhase; message?: string }) {
  if (phase === "unavailable") {
    return (
      <p className="rounded-md border border-line bg-panel px-3 py-2 text-xs text-muted">
        This deployment has no live backend configured, so custom plans can&rsquo;t be run
        here - only the precomputed cases on the home page work on this static site. Clone the
        repo and run it locally (see the README) to use the builder.
      </p>
    );
  }

  if (phase === "error") {
    return (
      <p className="rounded-md border border-bad/40 bg-bad/5 px-3 py-2 text-xs text-bad">
        {message ?? "something went wrong."}
      </p>
    );
  }

  if (phase === "waking") {
    return (
      <div className="rounded-md border border-accent/40 bg-accent/5 px-3 py-2 text-xs">
        <div className="flex items-center gap-2 text-ink">
          <PulseDot />
          <span>the live backend looks like it&rsquo;s asleep - waking it up</span>
        </div>
        <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-panel-2">
          <div className="h-full w-1/3 animate-[wake_1.6s_ease-in-out_infinite] rounded-full bg-accent" />
        </div>
        <p className="mt-1.5 text-muted">
          can take up to ~30s on a cold start - every other case on this site still works
          while you wait.
        </p>
        <style>{`@keyframes wake { 0% { margin-left: 0%; } 50% { margin-left: 67%; } 100% { margin-left: 0%; } }`}</style>
      </div>
    );
  }

  // "sending" - the common case, real physics running, not yet long enough to suspect sleep
  return (
    <div className="flex items-center gap-2 rounded-md border border-line bg-panel px-3 py-2 text-xs text-ink">
      <PulseDot />
      <span>{message ?? "running episode - real physics, not a canned animation…"}</span>
    </div>
  );
}
