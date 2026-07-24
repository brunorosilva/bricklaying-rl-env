"""PPO for the mobile-robot task (BrickLayerRobot-v0), hybrid action head.

Same CleanRL spine as train/ppo.py, but the policy is a HybridAgent (Categorical
mode + Gaussian offset/kind) and the rollout stores discrete + continuous actions
separately. Reuses compute_gae and the architecture registry, so `--arch` selects
any backbone.

    python -m train.ppo_robot --arch mlp --total-timesteps 2000000
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import atrium_sim  # noqa: F401
from atrium_sim.blueprint import _SUITES
from train.agent import HybridAgent, HybridAgentPolicy, save_hybrid_checkpoint
from train.ppo import compute_gae

N_MODES, BOX_DIM = 3, 2


@dataclass
class Args:
    exp_name: str = "robot"
    seed: int = 1
    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.02      # exploration bonus on the MOVE/PLACE head (keeps it
                                # from collapsing to always-move before it learns to place)
    ent_coef_box: float = 0.0   # none on the Gaussian offset/kind (avoids std runaway)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    arch: str = "mlp"
    torch_threads: int = 0
    device: str = "cpu"         # "cuda" to train on GPU (helps the matmul-bound
                                # transformer/cnn archs; MLPs stay env-bound so cpu is fine)
    random_start: bool = False  # False = base always starts at the left end (robot8): the
                                # leftmost-first staircase then completes in a pure rightward
                                # sweep, so MOVE_LEFT is never needed/learned. True (robot9)
                                # starts the base at a random point so work is sometimes to the
                                # LEFT -> reach-shaping rewards moving left -> learns to navigate
                                # both directions (robust to any start), not just sweep right.
    drop_control: bool = False  # box[1] chooses the release height (the arm homes at the wall
                                # top; the model lowers it before releasing). Impact velocity is
                                # an emergent consequence of the fall -> precision via physics.
    drop_penalty_frac: float = 0.0  # penalize the release height (~ impact energy) so the model
                                    # is pushed toward realistic gentle placement (drop mode only)
    prefill_prob: float = 0.0   # fraction of episodes that start with a random partial structure
                                # already built (the robot must complete a standing wall)
    fall_off_edge: bool = False  # driving off the end of the rail topples the gantry (episode ends)
    suite: str = "robot"        # small walls (3-5 modules): completable, so it learns to
                                # finish a course and stack levels (big walls are too long-horizon)
    eval_suite: str = "robot_eval"
    sigma_mm: float = 6.0       # sharp shoulder: precision within the reach window
    sigma_deg: float = 2.0
    eval_interval: int = 25
    eval_episodes: int = 9
    gif_every: int = 4
    run_dir: str = "runs"
    track: bool = False
    wandb_project: str = "atrium-sim"


def make_env(suite: str, sigma_mm: float, sigma_deg: float, random_start: bool,
             drop_control: bool = False, drop_penalty_frac: float = 0.0,
             prefill_prob: float = 0.0, fall_off_edge: bool = False):
    def thunk():
        env = gym.make("atrium_sim/BrickLayerRobot-v0")
        u = env.unwrapped
        # c_reach 4x: strongly reward moving toward the nearest open slot (either
        # direction). random_start controls whether work can be to the LEFT of the
        # base (forcing MOVE_LEFT to be learned) or always to the right (sweep only).
        # drop_control: box[1] chooses the release height (impact velocity is emergent);
        # drop_penalty_frac penalizes a high release toward gentle placement.
        # prefill_prob: some episodes start with a random partial structure to complete.
        u.env_cfg = type(u.env_cfg)(suite=suite, random_start=random_start, c_reach=2.0,
                                    drop_control=drop_control,
                                    drop_penalty_frac=drop_penalty_frac,
                                    prefill_prob=prefill_prob,
                                    fall_off_edge=fall_off_edge)
        # softer collapse/waste penalties so the agent is less "afraid" to attempt
        # the hard last bricks (top course, half-brick ends); precision plateau unchanged
        u.reward_cfg = type(u.reward_cfg)(
            sigma_mm=sigma_mm, sigma_deg=sigma_deg, collapse_penalty=0.5, c_waste=0.25
        )
        return env

    return thunk


def evaluate_robot(agent: HybridAgent, episodes: int, suite: str, sigma_mm: float,
                   sigma_deg: float, random_start: bool, drop_control: bool = False,
                   drop_penalty_frac: float = 0.0, prefill_prob: float = 0.0,
                   fall_off_edge: bool = False) -> dict:
    env = gym.make("atrium_sim/BrickLayerRobot-v0")
    u = env.unwrapped
    u.env_cfg = type(u.env_cfg)(suite=suite, random_start=random_start, c_reach=2.0,
                                drop_control=drop_control,
                                drop_penalty_frac=drop_penalty_frac,
                                prefill_prob=prefill_prob,
                                fall_off_edge=fall_off_edge)  # match training
    u.reward_cfg = type(u.reward_cfg)(sigma_mm=sigma_mm, sigma_deg=sigma_deg)
    policy = HybridAgentPolicy(agent)
    specs = _SUITES[suite]
    keys = ("episode_return", "frac_in_tol", "frac_filled", "completed", "moves", "placements")
    acc = {k: [] for k in keys}
    lowers = []  # drop_control diagnostic: mean decoded lower_frac on PLACE actions
    for i in range(episodes):
        obs, _ = env.reset(seed=10000 + i, options={"spec": specs[i % len(specs)]})
        done = False
        while not done:
            action = policy.act(obs)
            if int(action[0]) == 0:  # Mode.PLACE
                lowers.append((float(action[1][1]) + 1.0) / 2.0)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
        for k in keys:
            acc[k].append(info["metrics"][k])
    env.close()
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["lower_frac"] = float(np.mean(lowers)) if lowers else 0.0
    return out


def main(args: Args) -> dict:
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    run_name = f"{args.exp_name}_{args.arch}_s{args.seed}_{int(time.time())}"
    run_path = Path(args.run_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    if args.track:
        import wandb

        wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
                   sync_tensorboard=True)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(run_path))
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.suite, args.sigma_mm, args.sigma_deg, args.random_start,
                  args.drop_control, args.drop_penalty_frac, args.prefill_prob,
                  args.fall_off_edge)
         for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    dev = torch.device(args.device)
    agent = HybridAgent(obs_dim, N_MODES, BOX_DIM, arch=args.arch).to(dev)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=dev)
    modes_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=dev)
    boxes_buf = torch.zeros((args.num_steps, args.num_envs, BOX_DIM), device=dev)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=dev)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=dev)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=dev)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=dev)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=dev)
    last_losses: dict = {}
    last_eval: dict = {}
    n_evals = 0

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / num_updates) * args.learning_rate

        # --- ROLLOUT ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            with torch.no_grad():
                mode, box, logprob, _, _, value = agent.get_action_and_value(next_obs)
            modes_buf[step] = mode
            boxes_buf[step] = box
            logprobs_buf[step] = logprob
            values_buf[step] = value

            action = (mode.cpu().numpy().astype(np.int64), box.cpu().numpy().astype(np.float32))
            next_obs_np, reward, terminated, truncated, infos = envs.step(action)
            done = np.logical_or(terminated, truncated)
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=dev)
            dones_buf[step] = torch.as_tensor(done, dtype=torch.float32, device=dev)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=dev)

            trunc_only = np.logical_and(truncated, np.logical_not(terminated))
            if trunc_only.any():
                with torch.no_grad():
                    for i in np.flatnonzero(trunc_only):
                        fobs = torch.as_tensor(infos["final_obs"][i], dtype=torch.float32, device=dev)
                        rewards_buf[step, i] += args.gamma * agent.get_value(fobs.unsqueeze(0))[0]

            if "final_info" in infos:
                fi = infos["final_info"]["metrics"]
                for i in np.flatnonzero(fi["_episode_return"]):
                    writer.add_scalar("charts/episodic_return", fi["episode_return"][i], global_step)
                    for k in ("frac_in_tol", "frac_filled", "completed", "moves",
                              "placements", "invalid"):
                        writer.add_scalar(f"env/{k}", fi[k][i], global_step)

        # --- GAE ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
        advantages = compute_gae(rewards_buf, values_buf, dones_buf, next_value,
                                 args.gamma, args.gae_lambda)
        returns = advantages + values_buf

        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_modes = modes_buf.reshape(-1)
        b_boxes = boxes_buf.reshape(batch_size, BOX_DIM)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # --- UPDATE ---
        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start : start + minibatch_size]
                _, _, newlogprob, cat_ent, box_ent, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_modes[mb], b_boxes[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                if args.clip_vloss:
                    v_clipped = b_values[mb] + (newvalue - b_values[mb]).clamp(
                        -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max((newvalue - b_returns[mb]) ** 2,
                                             (v_clipped - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = args.ent_coef * cat_ent.mean() + args.ent_coef_box * box_ent.mean()
                loss = pg_loss - entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/mode_entropy", cat_ent.mean().item(), global_step)
        writer.add_scalar("losses/box_entropy", box_ent.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/pose_std", agent.pose_std_mm(), global_step)
        last_losses = {"policy_loss": pg_loss.item(), "value_loss": v_loss.item(),
                       "mode_entropy": cat_ent.mean().item(), "approx_kl": approx_kl.item()}

        if args.eval_interval and update % args.eval_interval == 0:
            last_eval = evaluate_robot(agent, args.eval_episodes, args.eval_suite, args.sigma_mm, args.sigma_deg, args.random_start, args.drop_control, args.drop_penalty_frac, args.prefill_prob, args.fall_off_edge)
            n_evals += 1
            for k, v in last_eval.items():
                writer.add_scalar(f"eval/{k}", v, global_step)
            save_hybrid_checkpoint(agent, str(run_path / "ckpt.pt"), extra={"args": vars(args)})
            if args.gif_every and n_evals % args.gif_every == 0:
                try:
                    from atrium_sim.render.recorder import record_episode

                    genv = gym.make("atrium_sim/BrickLayerRobot-v0", render_mode="rgb_array")
                    genv.unwrapped.env_cfg = type(genv.unwrapped.env_cfg)(
                        random_start=args.random_start, c_reach=2.0,
                        drop_control=args.drop_control,
                        drop_penalty_frac=args.drop_penalty_frac,
                        prefill_prob=args.prefill_prob,
                        fall_off_edge=args.fall_off_edge)
                    genv.unwrapped.reward_cfg = type(genv.unwrapped.reward_cfg)(
                        sigma_mm=args.sigma_mm, sigma_deg=args.sigma_deg)
                    record_episode(genv, HybridAgentPolicy(agent),
                                   str(run_path / f"eval_{global_step}.gif"), seed=10000)
                    genv.close()
                except Exception as e:
                    print(f"gif capture failed: {e}")
            print(f"update {update}/{num_updates}  step {global_step}  SPS {sps}  "
                  f"eval in-tol {last_eval.get('frac_in_tol', 0):.2%}  "
                  f"filled {last_eval.get('frac_filled', 0):.2%}  "
                  f"completed {last_eval.get('completed', 0):.2%}  "
                  f"return {last_eval.get('episode_return', 0):+.2f}", flush=True)

    save_hybrid_checkpoint(agent, str(run_path / "ckpt.pt"), extra={"args": vars(args)})
    envs.close()
    writer.close()
    return {"sps": sps, "losses": last_losses, "eval": last_eval,
            "ckpt": str(run_path / "ckpt.pt"), "global_step": global_step}


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
