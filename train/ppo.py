"""Self-contained PPO for atrium-sim, CleanRL-style.

Single file, readable top to bottom: rollout -> GAE -> clipped update.
The labelled sections (# --- ROLLOUT / GAE / UPDATE ---) are the GRPO seam:
grpo.py will be a copy-edit that replaces GAE + critic with group-normalised
episode returns over shared wall specs.

Two deliberate departures from stock CleanRL:
- gymnasium 1.x vector envs: we pin `AutoresetMode.SAME_STEP` and read
  `infos["final_info"]` as a dict-of-arrays-with-masks. The 0.x
  "final_observation" pattern silently corrupts GAE under 1.x defaults.
- NO observation/reward normalisation wrappers, ever: the episode return IS
  the audit score (see reward.py); wrapping it would destroy that identity.

    uv run python -m train.ppo --seed 1 --total-timesteps 5000000
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

import atrium_sim  # noqa: F401  (registers the env)
from train.agent import Agent, AgentPolicy, save_checkpoint


@dataclass
class Args:
    exp_name: str = "ppo"
    seed: int = 1
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128            # rollout length per env (batch = num_envs * num_steps)
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0           # entropy bonus PAYS a Gaussian to widen its std - run 1
                                    # (ent_coef=0.01) drove pose_std 0.61 -> 0.97 and pinned
                                    # precision at 0%; standard continuous-control default is 0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    # network architecture (see train/architectures.py: mlp, cnn, attention, ...)
    arch: str = "mlp"
    # "slot_relative" (env picks the slot, agent nudges ±15mm) or "absolute"
    # (agent places anywhere on the wall - the hard, free-placement variant).
    action_mode: str = "slot_relative"
    torch_threads: int = 0          # cap intra-op threads (>0) so a sweep can run in parallel
    # env / eval
    suite: str = "train"
    # Reward-shaping shoulder (does NOT touch the ±3mm tolerance / audit metric).
    # Matched to the tight ±15mm offset range: sharp enough that ≤3mm clearly
    # beats ~9mm (the exploration band), so precision pays; wide enough that the
    # whole ±15mm range gets graded reward (no gradient desert). sigma=12 was too
    # forgiving (9mm≈0.78) and the offset drifted; sigma=6 gives 9mm≈0.37, 3mm=1.
    sigma_mm: float = 6.0
    sigma_deg: float = 2.0
    collapse_terminal: bool = False  # topples are rare when placing at slots; harmless
    async_envs: bool = False        # AsyncVectorEnv for big campaigns
    eval_interval: int = 50         # updates between in-process evals (0 = off)
    eval_episodes: int = 10
    gif_every: int = 4              # record an eval GIF every N evals (0 = off)
    # io
    run_dir: str = "runs"
    track: bool = False             # wandb (sync_tensorboard)
    wandb_project: str = "atrium-sim"


def make_env(suite: str, sigma_mm: float, sigma_deg: float, collapse_terminal: bool,
             action_mode: str = "slot_relative"):
    def thunk():
        env = gym.make("atrium_sim/BrickLayer-v0")
        u = env.unwrapped
        u.env_cfg = type(u.env_cfg)(
            suite=suite, collapse_terminal=collapse_terminal, action_mode=action_mode
        )
        u.reward_cfg = type(u.reward_cfg)(sigma_mm=sigma_mm, sigma_deg=sigma_deg)
        return env

    return thunk


def compute_gae(
    rewards: torch.Tensor,   # (T, N)
    values: torch.Tensor,    # (T, N)
    dones: torch.Tensor,     # (T, N)  1.0 where the transition at step t was terminal
    next_value: torch.Tensor,  # (N,)
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """GAE(lambda). dones[t]=1 means episode ended AT step t, so obs[t+1] is a
    fresh reset obs (SAME_STEP autoreset) and must not be bootstrapped from."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nextvalue = values[t + 1] if t + 1 < T else next_value
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextvalue * nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages


def evaluate_inprocess(agent: Agent, episodes: int, suite: str = "interp",
                       action_mode: str = "slot_relative") -> dict:
    from train.evaluate import evaluate

    env = gym.make("atrium_sim/BrickLayer-v0")
    # eval must use the SAME action semantics the policy was trained with
    env.unwrapped.env_cfg = type(env.unwrapped.env_cfg)(action_mode=action_mode)
    result = evaluate(env, AgentPolicy(agent), suite, episodes)
    env.close()
    return {k: v["mean"] for k, v in result["metrics"].items()}


def main(args: Args) -> dict:
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    run_name = f"{args.exp_name}_s{args.seed}_{int(time.time())}"
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
        torch.set_num_threads(args.torch_threads)  # so a parallel sweep doesn't oversubscribe
    device = torch.device("cpu")

    vec_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vec_cls(
        [make_env(args.suite, args.sigma_mm, args.sigma_deg, args.collapse_terminal, args.action_mode)
         for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(obs_dim, act_dim, arch=args.arch).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim))
    actions_buf = torch.zeros((args.num_steps, args.num_envs, act_dim))
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs))
    rewards_buf = torch.zeros((args.num_steps, args.num_envs))
    dones_buf = torch.zeros((args.num_steps, args.num_envs))
    values_buf = torch.zeros((args.num_steps, args.num_envs))

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
    last_losses: dict = {}
    last_eval: dict = {}
    n_evals = 0

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # --- ROLLOUT ---------------------------------------------------------
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.numpy())
            done = np.logical_or(terminated, truncated)
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32)
            dones_buf[step] = torch.as_tensor(done, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)

            # truncation bootstrap: budget termination is terminal (no bootstrap),
            # but a TimeLimit truncation must bootstrap V(final_obs)
            trunc_only = np.logical_and(truncated, np.logical_not(terminated))
            if trunc_only.any():
                with torch.no_grad():
                    for i in np.flatnonzero(trunc_only):
                        fobs = torch.as_tensor(infos["final_obs"][i], dtype=torch.float32)
                        rewards_buf[step, i] += args.gamma * agent.get_value(fobs.unsqueeze(0))[0]

            if "final_info" in infos:
                fi = infos["final_info"]["metrics"]
                for i in np.flatnonzero(fi["_episode_return"]):
                    writer.add_scalar("charts/episodic_return", fi["episode_return"][i], global_step)
                    writer.add_scalar("charts/episodic_length", fi["placements"][i], global_step)
                    for k in ("frac_in_tol", "frac_filled", "waste_frac", "completed",
                              "mean_abs_dev_mm", "score"):
                        writer.add_scalar(f"env/{k}", fi[k][i], global_step)
                    for k in ("final_potential", "terminal_terms", "step_costs"):
                        writer.add_scalar(f"rewards/{k}", fi[k][i], global_step)

        # --- GAE --------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
        advantages = compute_gae(rewards_buf, values_buf, dones_buf, next_value,
                                 args.gamma, args.gae_lambda)
        returns = advantages + values_buf

        b_obs = obs_buf.reshape(batch_size, obs_dim)
        b_actions = actions_buf.reshape(batch_size, act_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # --- UPDATE -----------------------------------------------------------
        b_inds = np.arange(batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
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
                        -args.clip_coef, args.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        (newvalue - b_returns[mb]) ** 2,
                        (v_clipped - b_returns[mb]) ** 2,
                    ).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.numpy(), b_returns.numpy()
        var_y = np.var(y_true)
        explained_var = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/pose_std", agent.pose_std_mm(), global_step)
        last_losses = {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
        }

        if args.eval_interval and update % args.eval_interval == 0:
            last_eval = evaluate_inprocess(agent, args.eval_episodes, action_mode=args.action_mode)
            n_evals += 1
            for k, v in last_eval.items():
                writer.add_scalar(f"eval/{k}", v, global_step)
            save_checkpoint(agent, str(run_path / "ckpt.pt"), extra={"args": vars(args)})
            if args.gif_every and n_evals % args.gif_every == 0:
                try:
                    from atrium_sim.render.recorder import record_episode

                    genv = gym.make("atrium_sim/BrickLayer-v0", render_mode="rgb_array")
                    genv.unwrapped.env_cfg = type(genv.unwrapped.env_cfg)(action_mode=args.action_mode)
                    record_episode(genv, AgentPolicy(agent),
                                   str(run_path / f"eval_{global_step}.gif"), seed=10000)
                    genv.close()
                except Exception as e:  # rendering is optional; never kill training
                    print(f"gif capture failed: {e}")
            print(f"update {update}/{num_updates}  step {global_step}  SPS {sps}  "
                  f"eval in-tol {last_eval.get('frac_in_tol', 0):.2%}  "
                  f"return {last_eval.get('episode_return', 0):+.2f}")

    ckpt_path = str(run_path / "ckpt.pt")
    save_checkpoint(agent, ckpt_path, extra={"args": vars(args)})
    envs.close()
    writer.close()
    return {"sps": sps, "losses": last_losses, "eval": last_eval,
            "ckpt": ckpt_path, "global_step": global_step}


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
