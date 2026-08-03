import { NextRequest, NextResponse } from "next/server";
import { runEpisode } from "@/lib/runPython";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const { policy = "oracle", seed = 0, spec = "random", scenario = "empty",
            env = "bricklayer", plan } = await req.json();
    const args = [
      "--env", env === "robot" ? "robot" : "bricklayer",
      "--policy", String(policy),
      "--seed", String(seed),
      "--scenario", String(scenario),
    ];
    // a custom plan (the grid editor) is too big/structured for --spec's argv round-trip,
    // so it goes over stdin instead - either a full tiled FacadePlan or just
    // {grid_cols, grid_rows, openings}, which webviz.server tiles server-side either way.
    let stdinInput: string | undefined;
    if (plan) {
      args.push("--plan-stdin");
      stdinInput = JSON.stringify(plan);
    } else {
      args.push("--spec", String(spec));
    }
    const data = await runEpisode(args, stdinInput);
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
