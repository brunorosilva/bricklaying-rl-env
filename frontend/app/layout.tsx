import "./globals.css";
import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";

// Self-hosted at build time (next/font downloads once, bakes the woff2 into the static
// export under _next/static/media) - no runtime fetch to Google's CDN, so this works
// unchanged under GITHUB_PAGES's output:"export" and costs nothing on every page load.
const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"], display: "swap" });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "Bricklaying with RL",
  description: "Watch a policy lay a running-bond brick wall to BIM ±3mm tolerance.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="min-h-screen bg-bg font-sans text-[15px] text-ink antialiased">
        <header className="flex flex-wrap items-center gap-4 border-b border-line px-5 py-3.5">
          <Link href="/" className="text-[15px] font-semibold tracking-wide text-accent">
            Bricklaying with RL
          </Link>
          <nav className="flex gap-4 text-sm text-muted">
            <Link href="/" className="hover:text-ink">
              cases
            </Link>
            <Link href="/strike" className="hover:text-ink">
              the strike
            </Link>
            <Link href="/compare" className="hover:text-ink">
              compare
            </Link>
            <Link href="/build" className="hover:text-ink">
              build your own
            </Link>
          </nav>
          <span className="ml-auto text-xs text-muted">
            reward = live BIM audit (±3&nbsp;mm)
          </span>
        </header>
        {children}
      </body>
    </html>
  );
}
