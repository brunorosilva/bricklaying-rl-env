"""Record one episode as a compact, JSON-serialisable replay trajectory.

Unlike the GIF recorder (server-side rendered frames), this captures the raw
brick poses at every settle substep plus per-step reward/audit, so the browser
can replay the episode on a canvas with scrubbing and quality colouring - and
the payload stays a few hundred KB.
"""

from __future__ import annotations

from atrium_sim.blueprint import Blueprint, BrickKind, brick_face


def _targets_json(bp: Blueprint) -> list[dict]:
    out = []
    for t in bp.targets:
        w, h = brick_face(t.kind)
        out.append(
            {"tid": t.tid, "x": round(t.x, 1), "y": round(t.y, 1), "w": w, "h": h,
             "kind": int(t.kind), "course": t.course, "slot": t.slot}
        )
    return out


def _snapshot(world) -> list[list]:
    """One tick: every brick as [x, y, theta, kind, brick_id] (+ verts as a 6th element for
    VOUSSOIR bricks only). `brick_id` matches a step's `matches[].brick_id` - without it the
    frontend had no way to look up which brick a match belongs to except re-deriving the
    whole audit itself (a second, independently-maintained match_bricks that could silently
    drift from the real one - exactly what sending `matches` was meant to eliminate). A
    voussoir has no fixed (w, h) - it's a tapered wedge - so the frontend was drawing it as a
    plain rectangle (correct in the server-rendered GIFs, wrong in the browser); `verts` are
    LOCAL (body-frame, pre-rotation) polygon points, straight from physics.BrickPose - the
    frontend must rotate+translate them by (x, y, theta) itself, the same transform
    atrium_sim.render.renderer._poly_corners already applies. verts omitted (not null-
    padded) for flat bricks to keep every other tick's payload as small as possible."""
    out = []
    for p in world.poses():
        row = [round(p.x, 1), round(p.y, 1), round(p.theta, 4), int(p.kind), p.brick_id]
        if p.kind == BrickKind.VOUSSOIR and p.verts is not None:
            row.append([[round(vx, 1), round(vy, 1)] for vx, vy in p.verts])
        out.append(row)
    return out


def _matches_json(report) -> list[dict]:
    """The audit's own matches, per step - so the frontend's replay renders the SAME
    matching atrium_sim.reward computed, instead of re-deriving it client-side with a
    second, independently-maintained implementation of match_bricks that can silently
    drift from the real one (different GATE/TOL constants, a kind-filter bug, ...)."""
    return [
        {"brick_id": m.brick_id, "target_id": m.target_id, "dx": round(m.dx, 2),
         "dy": round(m.dy, 2), "d": round(m.d, 2), "in_tol": bool(m.in_tol)}
        for m in report.matches
    ]


def record_trajectory(env, policy, seed: int | None = None, spec=None, scenario: str = "empty") -> dict:
    """Roll one episode; return a replay dict (see the frontend for the schema)."""
    u = env.unwrapped
    flat_ticks: list[list[list]] = []
    try:
        options: dict = {"scenario": scenario}
        if spec is not None:
            options["spec"] = spec
        obs, info = env.reset(seed=seed, options=options)
        # attach the capture hook AFTER reset so scenario pre-fill settling isn't
        # recorded (prefilled bricks still appear in every frame - they're in the world)
        u.tick_callback = lambda: flat_ticks.append(_snapshot(u.world))
        steps = []
        done = False
        step_i = 0
        while not done:
            start = len(flat_ticks)
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # a step with no settle ticks (rare) still gets one snapshot so the
            # scrubber always advances
            ticks = flat_ticks[start:] or [_snapshot(u.world)]
            steps.append(
                {
                    "i": step_i,
                    "reward": round(float(reward), 4),
                    "return": round(float(u.episode_return), 4),
                    "cursor": int(u.cursor),
                    "frac_in_tol": round(u.report.frac_in_tol, 4),
                    "frac_filled": round(u.report.frac_filled, 4),
                    "waste": int(u.report.waste_count),
                    "matches": _matches_json(u.report),
                    "ticks": ticks,
                }
            )
            step_i += 1
    finally:
        u.tick_callback = None

    return {
        "spec": {"n_modules": u.blueprint.spec.n_modules, "n_courses": u.blueprint.n_courses},
        "length": round(u.blueprint.length, 1),
        "n_courses": u.blueprint.n_courses,
        "n_targets": u.blueprint.n_targets,
        "targets": _targets_json(u.blueprint),
        "steps": steps,
        "metrics": {k: round(float(v), 4) for k, v in info["metrics"].items()},
        "seed": seed,
    }


def record_robot_trajectory(env, policy, seed: int | None = None, spec=None, plan=None) -> dict:
    """Replay for the mobile robot: like record_trajectory, but also captures the base
    position each frame (so the canvas can draw the base + reach window sliding) and the mode
    (place/move) per step. A MOVE runs no physics, so it contributes one frame at the new base.

    A `plan` (FacadePlan) builds a whole facade/house instead of a single wall; the static hard
    bodies (cement arches, lintels, sills) are captured once with the frame index at which they
    first appear, so the canvas can fade them in as the build reaches them."""
    u = env.unwrapped
    flat: list[tuple[float, list]] = []  # (base_x, poses)
    hard: list[dict] = []                # static bodies, recorded once with their appear-frame
    seen: set[int] = set()

    def _capture_hard():
        for sid, kind, verts in u.world.hard_poses():
            if sid not in seen:
                seen.add(sid)
                hard.append({"kind": kind, "appear": max(0, len(flat) - 1),
                             "verts": [[round(x, 1), round(y, 1)] for x, y in verts]})

    def _tick():
        flat.append((round(u.base_x, 1), _snapshot(u.world)))
        _capture_hard()

    try:
        options = ({"plan": plan} if plan is not None
                   else ({"spec": spec} if spec is not None else None))
        obs, info = env.reset(seed=seed, options=options)
        u.tick_callback = _tick
        steps = []
        done = False
        step_i = 0
        while not done:
            start = len(flat)
            action = policy.act(obs)
            mode = int(action[0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if len(flat) == start:  # a move (no settle) -> one frame at the new base
                flat.append((round(u.base_x, 1), _snapshot(u.world)))
                _capture_hard()
            chunk = flat[start:]
            steps.append({
                "i": step_i, "mode": mode, "cursor": int(u._active_course()),
                "reward": round(float(reward), 4), "return": round(float(u.episode_return), 4),
                "frac_in_tol": round(u.report.frac_in_tol, 4),
                "frac_filled": round(u.report.frac_filled, 4),
                "matches": _matches_json(u.report),
                "moves": int(u.moves), "placements": int(u.placements),
                "base_ticks": [f[0] for f in chunk],
                "ticks": [f[1] for f in chunk],
            })
            step_i += 1
        _capture_hard()  # catch any hard bodies spawned in the final settle
    finally:
        u.tick_callback = None

    # a big panel can hit the gym step cap (truncation) before the env terminates, so
    # info has no "metrics" - rebuild them from the env's own terminal-info in that case
    metrics = info.get("metrics") or u._info(terminal=True)["metrics"]
    return {
        "spec": {"n_modules": u.blueprint.spec.n_modules, "n_courses": u.blueprint.n_courses},
        "length": round(u.blueprint.length, 1),
        "n_courses": u.blueprint.n_courses,
        "n_targets": u.blueprint.n_targets,
        "targets": _targets_json(u.blueprint),
        "robot": {"reach": u.env_cfg.reach_mm},
        "hard_bodies": hard,
        "steps": steps,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "seed": seed,
    }
