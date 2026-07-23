import { NextRequest, NextResponse } from "next/server";
import { runEpisode } from "@/lib/runPython";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const env = req.nextUrl.searchParams.get("env") === "robot" ? "robot" : "bricklayer";
  try {
    return NextResponse.json(await runEpisode(["--env", env, "--list"]));
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
