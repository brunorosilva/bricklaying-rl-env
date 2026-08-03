import "./globals.css";
import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";

export const metadata: Metadata = {
  title: "atrium-sim",
  description: "Watch a policy lay a running-bond brick wall to BIM ±3mm tolerance.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg font-sans text-[14px] text-ink antialiased">
        <header className="flex flex-wrap items-center gap-4 border-b border-line px-5 py-3.5">
          <Link href="/" className="text-[15px] font-semibold tracking-wide text-accent">
            atrium-sim
          </Link>
          <nav className="flex gap-4 text-sm text-muted">
            <Link href="/" className="hover:text-ink">
              cases
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
