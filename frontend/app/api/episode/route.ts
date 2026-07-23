import { NextRequest, NextResponse } from "next/server";
import { runEpisode } from "@/lib/runPython";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  try {
    const { policy = "oracle", seed = 0, spec = "random", scenario = "empty",
            env = "bricklayer" } = await req.json();
    const data = await runEpisode([
      "--env", env === "robot" ? "robot" : "bricklayer",
      "--policy", String(policy),
      "--seed", String(seed),
      "--spec", String(spec),
      "--scenario", String(scenario),
    ]);
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json({ error: (e as Error).message }, { status: 500 });
  }
}
