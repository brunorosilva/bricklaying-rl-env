"""Agent internals: distribution shapes, deterministic mode, GAE vs hand example."""

import numpy as np
import torch

from train.agent import Agent, CheckpointPolicy, save_checkpoint
from train.ppo import compute_gae

from atrium_sim.observations import OBS_DIM  # noqa: E402
ACT_DIM = 2


def test_action_and_value_shapes():
    agent = Agent(OBS_DIM, ACT_DIM)
    x = torch.randn(7, OBS_DIM)
    action, logprob, entropy, value = agent.get_action_and_value(x)
    assert action.shape == (7, ACT_DIM)
    assert logprob.shape == entropy.shape == value.shape == (7,)
    assert torch.isfinite(logprob).all() and (entropy > 0).all()
    # replaying the same actions must reproduce identical log-probs
    _, logprob2, _, _ = agent.get_action_and_value(x, action)
    assert torch.allclose(logprob, logprob2)


def test_deterministic_mode_is_mean_and_bounded():
    agent = Agent(OBS_DIM, ACT_DIM)
    obs = np.random.default_rng(0).uniform(-1, 1, OBS_DIM).astype(np.float32)
    a1, a2 = agent.act_deterministic(obs), agent.act_deterministic(obs)
    assert np.array_equal(a1, a2)
    assert np.all(a1 >= -1.0) and np.all(a1 <= 1.0)


def test_critic_false_for_grpo_seam():
    agent = Agent(OBS_DIM, ACT_DIM, critic=False)
    action, logprob, entropy, value = agent.get_action_and_value(torch.randn(3, OBS_DIM))
    assert value is None


def test_gae_matches_hand_computation():
    """T=4, gamma=0.9, lambda=0.8, terminal transition at t=2 (no bootstrap across it)."""
    rewards = torch.tensor([[1.0], [0.0], [2.0], [1.0]])
    values = torch.tensor([[0.5], [1.0], [0.2], [0.3]])
    dones = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    next_value = torch.tensor([0.4])
    adv = compute_gae(rewards, values, dones, next_value, gamma=0.9, gae_lambda=0.8)
    # hand: d3 = 1 + .9*.4 - .3 = 1.06 -> a3 = 1.06
    #       d2 = 2 - .2 = 1.8 (terminal) -> a2 = 1.8
    #       d1 = 0 + .9*.2 - 1 = -.82 -> a1 = -.82 + .72*1.8 = .476
    #       d0 = 1 + .9*1 - .5 = 1.4 -> a0 = 1.4 + .72*.476 = 1.74272
    expected = torch.tensor([[1.74272], [0.476], [1.8], [1.06]])
    assert torch.allclose(adv, expected, atol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    agent = Agent(OBS_DIM, ACT_DIM)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    expected = agent.act_deterministic(obs)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(agent, path, extra={"args": {"seed": 1}})
    policy = CheckpointPolicy(path)
    assert np.allclose(policy.act(obs), expected)
