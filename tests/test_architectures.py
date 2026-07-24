"""Every registered architecture must construct, forward, act, and roundtrip."""

import numpy as np
import pytest
import torch

from train.agent import Agent, CheckpointPolicy, save_checkpoint
from train.architectures import ARCHITECTURES

from atrium_sim.observations import OBS_DIM  # noqa: E402
ACT_DIM = 2


@pytest.mark.parametrize("arch", ARCHITECTURES)
def test_arch_forward_and_roundtrip(arch, tmp_path):
    agent = Agent(OBS_DIM, ACT_DIM, arch=arch)
    x = torch.randn(4, OBS_DIM)
    action, logprob, entropy, value = agent.get_action_and_value(x)
    assert action.shape == (4, ACT_DIM)
    assert logprob.shape == entropy.shape == value.shape == (4,)
    assert torch.isfinite(logprob).all() and torch.isfinite(value).all()

    det = agent.act_deterministic(np.zeros(OBS_DIM, dtype=np.float32))
    assert det.shape == (ACT_DIM,) and np.all(np.abs(det) <= 1.0)

    # checkpoint must preserve the architecture so it loads back correctly
    path = str(tmp_path / f"{arch}.pt")
    save_checkpoint(agent, path)
    loaded = CheckpointPolicy(path)
    assert loaded.agent.arch == arch
    assert np.allclose(loaded.act(np.zeros(OBS_DIM, dtype=np.float32)), det, atol=1e-5)


def test_critic_false_has_no_value_head():
    agent = Agent(OBS_DIM, ACT_DIM, arch="cnn", critic=False)
    _, _, _, value = agent.get_action_and_value(torch.randn(3, OBS_DIM))
    assert value is None
