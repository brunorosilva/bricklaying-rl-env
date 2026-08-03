"use client";

export default function GlobalError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main className="mx-auto max-w-2xl px-5 py-16 text-center">
      <h1 className="text-lg font-semibold text-bad">Something went wrong</h1>
      <p className="mt-2 text-sm text-muted">
        {error.message || "An unexpected error occurred while rendering this page."}
      </p>
      <button
        onClick={reset}
        className="mt-5 rounded-md border border-line bg-panel-2 px-4 py-2 text-sm text-ink hover:border-muted"
      >
        Try again
      </button>
    </main>
  );
}
