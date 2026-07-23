"""Actor-critic networks for BrickLayer, CleanRL-style.

Separate actor/critic backbones (no shared features - the obs is cheap to
encode twice and the value/policy gradients don't fight), each chosen from the
architecture registry (train/architectures.py). A linear Gaussian-mean head and
value head sit on top. Gaussian policy with a state-independent log-std; NO tanh
squash: actions are clipped at the env boundary and the log-prob is taken on the
unclipped sample.

`Agent(critic=False)` is the GRPO seam: group-normalised advantages need no
value head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from train.architectures import build_backbone, layer_init


class Agent(nn.Module):
    LOGSTD_INIT = -0.5
    LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, act_dim: int, arch: str = "mlp", critic: bool = True):
        super().__init__()
        self.obs_dim, self.act_dim, self.arch = obs_dim, act_dim, arch
        self.actor_backbone = build_backbone(arch, obs_dim)
        self.actor_head = layer_init(nn.Linear(self.actor_backbone.feat_dim, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), self.LOGSTD_INIT))
        if critic:
            self.critic_backbone = build_backbone(arch, obs_dim)
            self.critic_head = layer_init(nn.Linear(self.critic_backbone.feat_dim, 1), std=1.0)
        else:
            self.critic_backbone = None

    def actor_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.actor_backbone(x))

    def _dist(self, x: torch.Tensor) -> Normal:
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.clamp(self.LOGSTD_MIN, self.LOGSTD_MAX).expand_as(mean)
        return Normal(mean, logstd.exp())

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_backbone(x)).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        dist = self._dist(x)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(x) if self.critic_backbone is not None else None
        return action, logprob, entropy, value

    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """Eval mode: the mean action, clipped to the env's box.

        Switches to eval() so dropout/other train-only layers are inactive, then
        restores training mode (PPO updates need it).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a = self.actor_mean(x).squeeze(0).clamp(-1.0, 1.0).numpy()
        if was_training:
            self.train()
        return a

    def pose_std_mm(self) -> float:
        """Current x-dimension exploration std (normalised action units)."""
        return float(self.actor_logstd[0, 0].detach().exp())


class HybridAgent(nn.Module):
    """Actor-critic for the mobile robot's hybrid action: a Categorical head over
    the mode {place, move-left, move-right} plus a Gaussian head over the
    continuous [offset, kind]. Same backbone registry as `Agent`, so every
    architecture works unchanged. Joint log-prob = logp(mode) + sum logp(box)."""

    LOGSTD_INIT = -0.5
    LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, n_modes: int, box_dim: int, arch: str = "mlp",
                 critic: bool = True):
        super().__init__()
        self.obs_dim, self.n_modes, self.box_dim, self.arch = obs_dim, n_modes, box_dim, arch
        self.actor_backbone = build_backbone(arch, obs_dim)
        f = self.actor_backbone.feat_dim
        self.mode_head = layer_init(nn.Linear(f, n_modes), std=0.01)
        # Bias the initial mode distribution toward PLACE (index 0). Starting the
        # policy placing-greedily makes early experience place-heavy so it learns
        # "placing pays" + precision fast; movement is then discovered where it's
        # actually needed (stuck -> invalid-place penalty -> try MOVE). Without
        # this the untrained policy random-walks and collapses to always-move.
        self.mode_head.bias.data = torch.tensor([1.5] + [0.0] * (n_modes - 1))
        self.box_mean = layer_init(nn.Linear(f, box_dim), std=0.01)
        self.box_logstd = nn.Parameter(torch.full((1, box_dim), self.LOGSTD_INIT))
        if critic:
            self.critic_backbone = build_backbone(arch, obs_dim)
            self.critic_head = layer_init(nn.Linear(self.critic_backbone.feat_dim, 1), std=1.0)
        else:
            self.critic_backbone = None

    def _dists(self, x):
        f = self.actor_backbone(x)
        mean = self.box_mean(f)
        logstd = self.box_logstd.clamp(self.LOGSTD_MIN, self.LOGSTD_MAX).expand_as(mean)
        return Categorical(logits=self.mode_head(f)), Normal(mean, logstd.exp())

    def get_value(self, x):
        return self.critic_head(self.critic_backbone(x)).squeeze(-1)

    def get_action_and_value(self, x, mode=None, box=None):
        """Returns (mode, box, logprob, cat_entropy, box_entropy, value).

        The two entropies are kept separate so PPO can put an exploration bonus
        on the discrete move/place head WITHOUT inflating the Gaussian (an
        entropy bonus on the Gaussian drives a std runaway - see the base task)."""
        cat, normal = self._dists(x)
        if mode is None:
            mode = cat.sample()
        if box is None:
            box = normal.sample()
        logprob = cat.log_prob(mode) + normal.log_prob(box).sum(-1)
        value = self.get_value(x) if self.critic_backbone is not None else None
        return mode, box, logprob, cat.entropy(), normal.entropy().sum(-1), value

    def act_deterministic(self, obs: np.ndarray):
        """Eval: argmax mode + mean offset. Returns (mode:int, box:np.ndarray)."""
        was_training = self.training
        self.eval()
        dev = self.box_logstd.device  # match the agent's device (cpu or cuda)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            cat, normal = self._dists(x)
            mode = int(cat.logits.argmax(-1)[0])
            box = normal.mean.squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
        if was_training:
            self.train()
        return mode, box

    def pose_std_mm(self) -> float:
        return float(self.box_logstd[0, 0].detach().exp())


def save_hybrid_checkpoint(agent: HybridAgent, path: str, extra: dict | None = None) -> None:
    torch.save({
        "model_state_dict": agent.state_dict(), "obs_dim": agent.obs_dim,
        "n_modes": agent.n_modes, "box_dim": agent.box_dim, "arch": agent.arch,
        "hybrid": True, **(extra or {}),
    }, path)


def load_hybrid_agent(path: str) -> HybridAgent:
    # map to cpu so a cuda-trained checkpoint loads on cpu-only inference (frontend)
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    agent = HybridAgent(ckpt["obs_dim"], ckpt["n_modes"], ckpt["box_dim"],
                        arch=ckpt.get("arch", "mlp"))
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


class HybridAgentPolicy:
    """Policy-protocol wrapper: obs -> (mode, box) tuple action for the robot env."""

    def __init__(self, agent: HybridAgent):
        self.agent = agent

    def act(self, obs: np.ndarray):
        mode, box = self.agent.act_deterministic(obs)
        return (mode, box.astype(np.float32))


def save_checkpoint(agent: Agent, path: str, extra: dict | None = None) -> None:
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "obs_dim": agent.obs_dim,
            "act_dim": agent.act_dim,
            "arch": agent.arch,
            **(extra or {}),
        },
        path,
    )


def load_agent(path: str) -> Agent:
    ckpt = torch.load(path, weights_only=True)
    agent = Agent(ckpt["obs_dim"], ckpt["act_dim"], arch=ckpt.get("arch", "mlp"))
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


class CheckpointPolicy:
    """Policy-protocol wrapper for a saved checkpoint (used by train.evaluate)."""

    def __init__(self, path: str):
        self.agent = load_agent(path)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.agent.act_deterministic(obs)


class AgentPolicy:
    """Policy-protocol wrapper around a live agent (used for eval GIFs)."""

    def __init__(self, agent: Agent):
        self.agent = agent

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.agent.act_deterministic(obs)
