"""Reward audit: pinned worked example, plateau, folding, telescoping.

The worked example is the audit's contract: if any formula or constant
changes, these exact numbers change - and every published table with them.
"""

import math

import gymnasium as gym
import pytest

import atrium_sim  # noqa: F401
from atrium_sim.blueprint import Blueprint, BrickKind, BrickTarget, WallSpec
from atrium_sim.physics import BrickPose
from atrium_sim.reward import RewardConfig, audit, fold_angle, plateau_gauss, potential

CFG = RewardConfig()


def synthetic_blueprint(n: int = 20) -> Blueprint:
    """N same-kind targets in one row, 220mm apart (respects the 120mm invariant)."""
    targets = tuple(
        BrickTarget(tid=i, course=0, slot=i, x=105.0 + 220.0 * i, y=30.0, kind=BrickKind.FULL)
        for i in range(n)
    )
    return Blueprint(
        spec=WallSpec(n_modules=n, n_courses=1),
        length=220.0 * n - 10.0,
        targets=targets,
        _courses=(targets,),
    )


def brick(i: int, dx: float = 0.0, dy: float = 0.0, deg: float = 0.0) -> BrickPose:
    return BrickPose(i, 105.0 + 220.0 * i + dx, 30.0 + dy, math.radians(deg), BrickKind.FULL)


def phi(bricks, bp, **kw) -> float:
    return potential(audit(bricks, bp, CFG, **kw), CFG)


class TestWorkedExample:
    """N=20, defaults: r_scale/N = 0.5 per perfect brick, step cost 0.01."""

    bp = synthetic_blueprint(20)
    step_cost = CFG.c_step_frac * CFG.r_scale / 20  # 0.01

    def test_step1_brick_2mm_off(self):
        # 2mm off, 0.2 deg tilt: both on the plateau -> q = 1.0
        r1 = phi([brick(0, dx=2.0, deg=0.2)], self.bp) - 0.0 - self.step_cost
        assert r1 == pytest.approx(0.490, abs=1e-3)

    def test_step2_brick_8mm_off(self):
        # 8mm off, 1.5 deg tilt: q = 0.84062 * 0.77880 = 0.65468
        s1 = [brick(0, dx=2.0, deg=0.2)]
        s2 = s1 + [brick(1, dx=8.0, deg=1.5)]
        r2 = phi(s2, self.bp) - phi(s1, self.bp) - self.step_cost
        assert r2 == pytest.approx(0.317, abs=1e-3)

    def test_step3_toppled_brick_is_stray(self):
        # toppled: 130mm from target at 78 deg -> outside gate -> stray, waste
        s2 = [brick(0, dx=2.0, deg=0.2), brick(1, dx=8.0, deg=1.5)]
        s3 = s2 + [brick(2, dx=130.0, deg=78.0)]
        r3 = phi(s3, self.bp) - phi(s2, self.bp) - self.step_cost
        assert r3 == pytest.approx(-0.260, abs=1e-3)

    def test_step3_variant_knocks_neighbour(self):
        # topple also knocks brick B from 8mm to 20mm: claw-back is automatic
        s2 = [brick(0, dx=2.0, deg=0.2), brick(1, dx=8.0, deg=1.5)]
        s3 = [brick(0, dx=2.0, deg=0.2), brick(1, dx=20.0, deg=1.5), brick(2, dx=130.0, deg=78.0)]
        r3v = phi(s3, self.bp) - phi(s2, self.bp) - self.step_cost
        assert r3v == pytest.approx(-0.535, abs=1e-3)

    def test_running_total_telescopes(self):
        s3 = [brick(0, dx=2.0, deg=0.2), brick(1, dx=8.0, deg=1.5), brick(2, dx=130.0, deg=78.0)]
        assert phi(s3, self.bp) - 3 * self.step_cost == pytest.approx(0.547, abs=1e-3)


class TestFormulas:
    def test_plateau_is_flat_inside_tolerance(self):
        assert plateau_gauss(0.0, 3.0, 12.0) == plateau_gauss(2.9, 3.0, 12.0) == 1.0

    def test_gaussian_reference_values(self):
        assert plateau_gauss(8.0, 3.0, 12.0) == pytest.approx(0.84062, abs=1e-4)
        assert plateau_gauss(15.0, 3.0, 12.0) == pytest.approx(0.36788, abs=1e-4)

    def test_angle_folding_flipped_brick_scores_perfect(self):
        # a brick that tumbled and landed flat-but-flipped (angle ~ pi) is
        # geometrically perfect: it must match with q ~ 1, not block the slot
        bp = synthetic_blueprint(2)
        flipped = BrickPose(0, 105.0, 30.0, math.pi + math.radians(0.1), BrickKind.FULL)
        report = audit([flipped], bp, CFG)
        assert len(report.matches) == 1
        assert report.matches[0].q == pytest.approx(1.0)
        assert abs(fold_angle(math.pi + 0.3)) < math.pi / 2

    def test_kind_gate_half_cannot_fill_full_slot(self):
        bp = synthetic_blueprint(2)
        half = BrickPose(0, 105.0, 30.0, 0.0, BrickKind.HALF)
        report = audit([half], bp, CFG, halves_used=1)
        assert not report.matches
        assert report.stray_bricks == (0,)
        assert report.waste_count == 2  # stray + unnecessary cut

    def test_second_brick_on_same_target_is_stray(self):
        bp = synthetic_blueprint(2)
        report = audit([brick(0), brick(7, dx=-1540.0 + 5.0)], bp, CFG)  # both near target 0
        assert len(report.matches) == 1
        assert report.matches[0].brick_id == 0  # nearer one wins
        assert len(report.stray_bricks) == 1


class TestTelescoping:
    """Env wiring: sum of step rewards must equal final potential + terminal
    terms - step costs, for arbitrary random rollouts. Catches double-counted
    or skipped audits in every termination path."""

    @pytest.mark.parametrize("seed", range(12))
    def test_random_rollout_return_identity(self, seed):
        env = gym.make("atrium_sim/BrickLayer-v0")
        env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(suite="train")
        obs, _ = env.reset(seed=seed, options={"spec": WallSpec(4, 2)})
        env.action_space.seed(seed)
        total, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            total += r
            done = terminated or truncated
        m = info["metrics"]
        expected = m["final_potential"] + m["terminal_terms"] - m["step_costs"]
        assert total == pytest.approx(expected, abs=1e-9)
        assert m["episode_return"] == pytest.approx(total, abs=1e-9)
        env.close()

    def test_oracle_like_rollout_scores_high(self):
        """Placing every brick exactly at its target must approach score 1.0
        (works in either action mode via the mode-aware oracle)."""
        from baselines.oracle import OraclePolicy

        env = gym.make("atrium_sim/BrickLayer-v0")
        obs, _ = env.reset(seed=0, options={"spec": WallSpec(4, 2)})
        policy = OraclePolicy(env)
        total, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, info = env.step(policy.act(obs))
            total += r
            done = terminated or truncated
        m = info["metrics"]
        assert m["frac_filled"] == 1.0
        assert m["frac_in_tol"] >= 0.9
        assert total > 10.0  # ~ r_scale + bonuses - costs
        env.close()
