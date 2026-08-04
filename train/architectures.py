"""A registry of policy/value backbones for the architecture bake-off.

Every backbone maps a flat observation to a feature vector; `Agent`/`HybridAgent`
(train/agent.py) put a Gaussian-mean head and a value head on top. Two unrelated
tasks share this registry, and "cnn"/"attention"/"attention2" mean something
different for each - dispatched in build_backbone() by obs_dim, since there's no
other signal available at construction time:

- The base bricklayer task (538-d, see atrium_sim/observations.py): obs[:528] is a
  real (course x slot x feature) grid (C_MAX=6, S_MAX=11, N_SLOT_FEATURES=8) + 10
  global scalars. CNN/SlotAttention split it back into that grid - the spatial
  structure a flat MLP would otherwise have to rediscover.
- The mobile-robot task (28-d, see atrium_sim.envs.robot_env._obs): a flat SENSOR
  vector (rail position, nearest-target direction, placement feedback, action
  mask, ...) with no grid at all - that grid was deliberately dropped (a prior
  refactor moved this task off the blueprint-grid observation entirely). FlatCNN/
  FlatAttention instead treat the 28 scalars as a plain sequence (1D conv) or as
  individual tokens via per-feature learned tokenization (see _FeatureTokenizer -
  the numeric-feature-tokenization scheme from FT-Transformer), since there's no
  spatial layout to exploit, only adjacency-in-the-vector and feature identity.
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
    "cnn",           # conv stack -> pool -> 2-layer dense head over the grid
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
    """Conv feature-extractor over the (course x slot) brick grid, then a proper
    multi-layer dense head (with the global scalars fused in) - conv -> flatten ->
    dense-head, instead of the old conv -> single-Linear projection.

    NO spatial pooling: the task hinges on *which specific slot* is the next target,
    so pooling (which averages per-slot resolution away) is a poor fit here - the
    convs stay padded to keep the full 6x11 grid, and every cell reaches the dense
    head. Conv blocks (3x3, 8->32->32) read local brick neighborhoods; the two dense
    layers do the reasoning."""

    def __init__(self, n_glob=N_GLOB, hidden=128, channels=32):
        super().__init__()
        self.split = _SlotSplit()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(N_SLOT_FEATURES, channels, 3, padding=1)), nn.Tanh(),
            layer_init(nn.Conv2d(channels, channels, 3, padding=1)), nn.Tanh(),
        )
        conv_out = channels * C_MAX * S_MAX
        self.head = nn.Sequential(
            layer_init(nn.Linear(conv_out + n_glob, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, hidden)), nn.Tanh(),
        )
        self.feat_dim = hidden

    def forward(self, x):
        slots, glob = self.split(x)
        c = self.conv(slots.permute(0, 3, 1, 2))  # (B, F, C, S), grid preserved
        c = c.reshape(c.shape[0], -1)
        return self.head(torch.cat([c, glob], dim=1))


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


class _FeatureTokenizer(nn.Module):
    """Each flat obs scalar becomes its own token via a per-feature learned affine map:
    token_i = x_i * w_i + b_i (numeric feature tokenization, see FT-Transformer). Lets a
    transformer self-attend across raw sensor scalars that have no spatial grid between
    them, only a fixed identity (index i is always "nearest_dx", index 20 is always
    "next_voussoir", etc) - w_i/b_i give each feature its own learned embedding direction."""

    def __init__(self, n: int, d_model: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, d_model) * 0.02)
        self.b = nn.Parameter(torch.zeros(n, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.w + self.b  # (B, n, d_model)


class FlatAttention(nn.Module):
    """SlotAttention's analogue for a flat sensor observation with no (course x slot)
    grid to split - tokenizes each scalar (_FeatureTokenizer), self-attends, mean-pools."""

    def __init__(self, obs_dim: int, d_model=64, nhead=4, layers=1, hidden=128):
        super().__init__()
        self.tokenize = _FeatureTokenizer(obs_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, batch_first=True, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc, layers)
        self.fc = nn.Sequential(layer_init(nn.Linear(d_model, hidden)), nn.Tanh())
        self.feat_dim = hidden

    def forward(self, x):
        h = self.tr(self.tokenize(x)).mean(dim=1)
        return self.fc(h)


class FlatCNN(nn.Module):
    """CNN's analogue for a flat sensor observation with no (course x slot) grid - 1D
    convs read local neighborhoods among ADJACENT obs indices instead (robot_env._obs
    groups related sensors together - rail position, then work-sensing, then placement
    feedback, ... - so nearby indices are usually semantically related, just not spatial
    in the image sense), then a dense head."""

    def __init__(self, obs_dim: int, hidden=128, channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv1d(1, channels, 3, padding=1)), nn.Tanh(),
            layer_init(nn.Conv1d(channels, channels, 3, padding=1)), nn.Tanh(),
        )
        self.head = nn.Sequential(
            layer_init(nn.Linear(channels * obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, hidden)), nn.Tanh(),
        )
        self.feat_dim = hidden

    def forward(self, x):
        c = self.conv(x.unsqueeze(1))  # (B, 1, obs_dim) -> (B, channels, obs_dim)
        return self.head(c.reshape(c.shape[0], -1))


BASE_TASK_OBS_DIM = SLOT_DIM + N_GLOB  # 538 - the only obs_dim with a real slot grid


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
    if obs_dim == BASE_TASK_OBS_DIM:
        # real (course x slot) grid + globals; _SlotSplit slices the grid off the front
        n_glob = obs_dim - SLOT_DIM
        if name == "cnn":
            return CNN(n_glob, hidden=128)
        if name == "attention":
            return SlotAttention(n_glob, 64, 4, 1, 128)
        if name == "attention2":
            return SlotAttention(n_glob, 64, 4, 2, 128)
    else:
        # flat sensor vector (the robot task), no grid - see FlatCNN/FlatAttention
        if name == "cnn":
            return FlatCNN(obs_dim, hidden=128)
        if name == "attention":
            return FlatAttention(obs_dim, 64, 4, 1, 128)
        if name == "attention2":
            return FlatAttention(obs_dim, 64, 4, 2, 128)
    raise ValueError(f"unknown architecture: {name!r} (choices: {ARCHITECTURES})")
