/** @type {import('next').NextConfig} */
// GITHUB_PAGES=true (set by .github/workflows/pages.yml) builds this as a static export
// served from https://brunorosilva.github.io/bricklaying-rl-env/ - a GitHub *project*
// page, not a user page, so every internal path needs the repo-name prefix. basePath is
// the single source of truth for that prefix; NEXT_PUBLIC_BASE_PATH is derived from it
// (not set independently) so the two can never drift out of sync - client code that has
// to hand-prefix a fetch URL (frontend/lib/traces.ts) reads the env var, everything else
// (next/link, next/image, _next/* asset URLs) gets it for free from `basePath` itself.
const basePath = process.env.GITHUB_PAGES ? "/bricklaying-rl-env" : "";

const nextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  images: { unoptimized: true },
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
