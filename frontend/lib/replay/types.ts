// Mirrors webviz/trajectory.py's JSON output exactly - see that file for the producer side.

/** [x, y, theta(rad), kind(0=full,1=half,2=voussoir), brick_id, verts?] - `brick_id` matches
 * a Match's brick_id; `verts` (local, pre-rotation polygon points) are present ONLY for
 * kind===2 (a voussoir has no fixed w/h). */
export type Brick =
  | [number, number, number, number, number]
  | [number, number, number, number, number, [number, number][]];

export type Target = {
  tid: number; x: number; y: number; w: number; h: number; kind: number; course: number; slot: number;
};

/** The audit's own match, one per matched brick - sent by the server so the frontend never
 * re-derives matching itself (see trajectory.py's _matches_json). */
export type Match = { brick_id: number; target_id: number; dx: number; dy: number; d: number; in_tol: boolean };

export type Step = {
  i: number;
  reward: number;
  return: number;
  cursor?: number;
  frac_in_tol: number;
  frac_filled: number;
  waste?: number;
  matches: Match[];
  ticks: Brick[][];
  // robot-only
  mode?: number;
  base_ticks?: number[];
  moves?: number;
  placements?: number;
};

export type Metrics = {
  frac_in_tol: number;
  frac_filled: number;
  waste_count: number;
  episode_return: number;
  mean_abs_dev_mm: number;
  placements: number;
  moves?: number;
  ring_closure?: number;
  arch_strike_survival?: number;
  deadlocked?: number;
};

/** `disappear` is set only for a struck arch's centering (the timber former genuinely
 * leaves the physics world when the ring closes - see atrium_sim/envs/robot_env.py's
 * _strike_arch and webviz/trajectory.py's _capture_hard) - absent means "never removed"
 * (lintels, sills, permanent skewback stone). */
export type HardBody = { kind: string; appear: number; disappear?: number; verts: [number, number][] };

/** One real structural arch (semicircular/segmental/jack), in GLOBAL mm - not per-tick data,
 * just enough to group voussoir bricks and hard_bodies by which arch they belong to (by
 * x-range: `x0 <= x <= x1`), since a voussoir's dynamic BrickPose carries no arch identity
 * once spawned (see webviz/trajectory.py's own comment on why). Used by the "Strike" page. */
export type ArchRegion = { index: number; style: string; x0: number; x1: number; spring_y: number };

export type Replay = {
  spec: { n_modules: number; n_courses: number };
  length: number;
  n_courses: number;
  n_targets: number;
  targets: Target[];
  steps: Step[];
  metrics: Metrics;
  seed: number | null;
  truncated?: boolean;
  _policy?: string;
  robot?: { reach: number };
  hard_bodies?: HardBody[];
  arch_regions?: ArchRegion[];
};

/** One flattened animation frame - a single physics tick, with its owning step. */
export type Frame = { st: Step; bricks: Brick[]; base?: number; gi: number };

export type View = { xmin: number; ymax: number; s: number; ox: number; oy: number };
