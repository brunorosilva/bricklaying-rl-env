import { execFile } from "node:child_process";
import path from "node:path";

// `npm run dev` runs from frontend/, so the repo root (which contains .venv and
// the webviz package) is the parent directory. Override with ATRIUM_REPO_ROOT.
const REPO_ROOT = process.env.ATRIUM_REPO_ROOT || path.resolve(process.cwd(), "..");
const PYTHON = path.join(REPO_ROOT, ".venv", "bin", "python");

/**
 * Spawn a fresh Python process to run `python -m webviz.episode <args>` and
 * parse its JSON stdout. A new process per call means every episode runs the
 * CURRENT env code - no long-lived server to go stale.
 *
 * `stdinInput`, if given, is written to the child's stdin then the stream is
 * closed - pairs with `--plan-stdin` for a custom (grid-editor-built) plan,
 * which is too big/structured to round-trip through argv the way --spec does.
 * Stdin is always closed (even with no input) so a flag-less run never blocks
 * waiting on a read that will never happen.
 */
export function runEpisode(args: string[], stdinInput?: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const child = execFile(
      PYTHON,
      ["-m", "webviz.episode", ...args],
      { cwd: REPO_ROOT, maxBuffer: 64 * 1024 * 1024, timeout: 60_000 },
      (err, stdout, stderr) => {
        if (err) return reject(new Error(stderr?.trim() || err.message));
        try {
          resolve(JSON.parse(stdout));
        } catch {
          reject(new Error("unparseable output from python: " + stdout.slice(0, 300)));
        }
      },
    );
    child.stdin?.end(stdinInput);
  });
}
