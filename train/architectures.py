"""A registry of policy/value backbones for the architecture bake-off.

Every backbone maps the flat 538-d observation to a feature vector; `Agent`
(train/agent.py) puts a linear Gaussian-mean head and a linear value head on
top. The spatial backbones (CNN, attention) split the observation back into its
(course x slot x feature) grid + the global scalars - the structure the flat
MLP has to rediscover.

The observation layout (see atrium_sim/observations.py):
  obs[:528]  -> slot tensor  (C_MAX=6, S_MAX=11, N_SLOT_FEATURES=8)
  obs[528:]  -> 10 global scalars
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from atrium_sim.constants import C_MAX, S_MAX
from atrium_sim.observations import N_GLOBALS, N_SLOT_FEATURES

SLOT_DIM = C_MAX * S_MAX * N_SLOT_FEATURES  # 528
N_GLOB = N_GLOBALS                          # 10

ARCHITECTURES = [
    "mlp",           # baseline: 2x128 tanh
    "mlp_wide",      # 2x256 tanh
    "mlp_deep",      # 3x128 tanh
    "relu_wide",     # 2x256 relu (activation variant)
    "mlp_dropout",   # 2x128 tanh + dropout 0.1
    "mlp_layernorm", # 2x128 tanh + layernorm
    "resmlp",        # input proj + 2 residual blocks
    "cnn",           # conv over the (feature, course, slot) grid
    "attention",     # 1-layer transformer over the 66 slot tokens
    "attention2",    # 2-layer transformer
]


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _SlotSplit(nn.Module):
    """Flat obs -> (slots (B, C, S, F), globals (B, G))."""

    def forward(self, x: torch.Tensor):
        slots = x[:, :SLOT_DIM].reshape(-1, C_MAX, S_MAX, N_SLOT_FEATURES)
        return slots, x[:, SLOT_DIM:]


class MLP(nn.Module):
    def __init__(self, obs_dim, hidden, depth, act=nn.Tanh, dropout=0.0, layernorm=False):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for _ in range(depth):
            layers.append(layer_init(nn.Linear(d, hidden)))
            if layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.net = nn.Sequential(*layers)
        self.feat_dim = hidden

    def forward(self, x):
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1 = layer_init(nn.Linear(h, h))
        self.l2 = layer_init(nn.Linear(h, h))
        self.act = nn.Tanh()

    def forward(self, x):
        return x + self.l2(self.act(self.l1(x)))


class ResMLP(nn.Module):
    def __init__(self, obs_dim, hidden=128, blocks=2):
        super().__init__()
        self.proj = layer_init(nn.Linear(obs_dim, hidden))
        self.act = nn.Tanh()
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(blocks)])
        self.feat_dim = hidden

    def forward(self, x):
        h = self.act(self.proj(x))
        for b in self.blocks:
            h = b(h)
        return h


class CNN(nn.Module):
    def __init__(self, n_glob=N_GLOB, hidden=128, channels=32):
        super().__init__()
        self.split = _SlotSplit()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(N_SLOT_FEATURES, channels, 3, padding=1)), nn.Tanh(),
            layer_init(nn.Conv2d(channels, channels, 3, padding=1)), nn.Tanh(),
        )
        self.fc = nn.Sequential(
            layer_init(nn.Linear(channels * C_MAX * S_MAX + n_glob, hidden)), nn.Tanh(),
        )
        self.feat_dim = hidden

    def forward(self, x):
        slots, glob = self.split(x)
        c = self.conv(slots.permute(0, 3, 1, 2))  # (B, F, C, S)
        c = c.reshape(c.shape[0], -1)
        return self.fc(torch.cat([c, glob], dim=1))


class SlotAttention(nn.Module):
    """Treat the 66 slots as tokens, self-attend, mean-pool, fuse globals."""

    def __init__(self, n_glob=N_GLOB, d_model=64, nhead=4, layers=1, hidden=128):
        super().__init__()
        self.split = _SlotSplit()
        self.embed = layer_init(nn.Linear(N_SLOT_FEATURES, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, batch_first=True, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc, layers)
        self.fc = nn.Sequential(layer_init(nn.Linear(d_model + n_glob, hidden)), nn.Tanh())
        self.feat_dim = hidden

    def forward(self, x):
        slots, glob = self.split(x)
        tokens = slots.reshape(slots.shape[0], C_MAX * S_MAX, N_SLOT_FEATURES)
        h = self.tr(self.embed(tokens)).mean(dim=1)  # (B, d_model)
        return self.fc(torch.cat([h, glob], dim=1))


def build_backbone(name: str, obs_dim: int) -> nn.Module:
    if name == "mlp":
        return MLP(obs_dim, 128, 2)
    if name == "mlp_wide":
        return MLP(obs_dim, 256, 2)
    if name == "mlp_deep":
        return MLP(obs_dim, 128, 3)
    if name == "relu_wide":
        return MLP(obs_dim, 256, 2, act=nn.ReLU)
    if name == "mlp_dropout":
        return MLP(obs_dim, 128, 2, dropout=0.1)
    if name == "mlp_layernorm":
        return MLP(obs_dim, 128, 2, layernorm=True)
    if name == "resmlp":
        return ResMLP(obs_dim, 128, 2)
    # spatial backbones need the actual global-feature count (base env = 10,
    # robot env = 16); _SlotSplit already slices the slot grid off the front
    n_glob = obs_dim - SLOT_DIM
    if name == "cnn":
        return CNN(n_glob, hidden=128)
    if name == "attention":
        return SlotAttention(n_glob, 64, 4, 1, 128)
    if name == "attention2":
        return SlotAttention(n_glob, 64, 4, 2, 128)
    raise ValueError(f"unknown architecture: {name!r} (choices: {ARCHITECTURES})")
