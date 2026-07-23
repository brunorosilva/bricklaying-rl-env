"""Record one episode as a compact, JSON-serialisable replay trajectory.

Unlike the GIF recorder (server-side rendered frames), this captures the raw
brick poses at every settle substep plus per-step reward/audit, so the browser
can replay the episode on a canvas with scrubbing and quality colouring - and
the payload stays a few hundred KB.
"""

from __future__ import annotations

from atrium_sim.blueprint import Blueprint, brick_face


def _targets_json(bp: Blueprint) -> list[dict]:
    out = []
    for t in bp.targets:
        w, h = brick_face(t.kind)
        out.append(
            {"x": round(t.x, 1), "y": round(t.y, 1), "w": w, "h": h,
             "kind": int(t.kind), "course": t.course, "slot": t.slot}
        )
    return out


def _snapshot(world) -> list[list]:
    """One tick: every brick as [x, y, theta, kind], rounded to keep JSON small."""
    return [
        [round(p.x, 1), round(p.y, 1), round(p.theta, 4), int(p.kind)]
        for p in world.poses()
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


def record_robot_trajectory(env, policy, seed: int | None = None, spec=None) -> dict:
    """Replay for the mobile robot: like record_trajectory, but also captures the
    base position each frame (so the canvas can draw the base + reach window
    sliding) and the mode (place/move) per step. A MOVE runs no physics, so it
    contributes one frame at the new base position."""
    u = env.unwrapped
    flat: list[tuple[float, list]] = []  # (base_x, poses)
    try:
        obs, info = env.reset(seed=seed, options={"spec": spec} if spec is not None else None)
        u.tick_callback = lambda: flat.append((round(u.base_x, 1), _snapshot(u.world)))
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
            chunk = flat[start:]
            steps.append({
                "i": step_i, "mode": mode,
                "reward": round(float(reward), 4), "return": round(float(u.episode_return), 4),
                "frac_in_tol": round(u.report.frac_in_tol, 4),
                "frac_filled": round(u.report.frac_filled, 4),
                "moves": int(u.moves), "placements": int(u.placements),
                "base_ticks": [f[0] for f in chunk],
                "ticks": [f[1] for f in chunk],
            })
            step_i += 1
    finally:
        u.tick_callback = None

    return {
        "spec": {"n_modules": u.blueprint.spec.n_modules, "n_courses": u.blueprint.n_courses},
        "length": round(u.blueprint.length, 1),
        "n_courses": u.blueprint.n_courses,
        "n_targets": u.blueprint.n_targets,
        "targets": _targets_json(u.blueprint),
        "robot": {"reach": u.env_cfg.reach_mm},
        "steps": steps,
        "metrics": {k: round(float(v), 4) for k, v in info["metrics"].items()},
        "seed": seed,
    }
