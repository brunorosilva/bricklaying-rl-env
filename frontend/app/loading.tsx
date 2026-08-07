export default function Loading() {
  return (
    <main>
      <div className="h-[72vh] min-h-[440px] w-full animate-pulse bg-panel md:h-[82vh]" />
      <div className="mx-auto max-w-6xl px-5 py-10">
        <div className="mb-6 h-8 w-64 animate-pulse rounded-md bg-panel-2" />
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="h-32 animate-pulse rounded-lg border border-line bg-panel" />
          ))}
        </div>
      </div>
    </main>
  );
}
