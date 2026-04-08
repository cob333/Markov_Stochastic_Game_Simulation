from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wireless_marl.env import EnvParams, WirelessMarkovGame
from wireless_marl.utils import clip01, topology_adjacency


def test_clip01_bounds() -> None:
    assert clip01(-1.0) == 0.0
    assert clip01(0.5) == 0.5
    assert clip01(2.0) == 1.0


def test_topology_shapes_and_diagonal() -> None:
    for name in ("all_to_all", "star", "ring"):
        adjacency = topology_adjacency(4, name)
        assert adjacency.shape == (4, 4)
        assert np.all(np.diag(adjacency) == 0)


def test_reset_and_step_shapes() -> None:
    env = WirelessMarkovGame(EnvParams())
    obs = env.reset(seed=7)
    assert set(obs) == {0, 1, 2, 3}
    next_obs, rewards, terminated, truncated, info = env.step({i: 0 for i in range(4)})
    assert set(next_obs) == {0, 1, 2, 3}
    assert set(rewards) == {0, 1, 2, 3}
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "p_congest_next" in info


def test_seed_reproducibility() -> None:
    env_a = WirelessMarkovGame(EnvParams())
    env_b = WirelessMarkovGame(EnvParams())
    obs_a = env_a.reset(seed=123)
    obs_b = env_b.reset(seed=123)
    for agent_id in obs_a:
        assert np.array_equal(obs_a[agent_id], obs_b[agent_id])


def test_collision_occurs_for_two_transmitters() -> None:
    env = WirelessMarkovGame(EnvParams())
    env.reset(seed=1)
    env.buffers[:] = 1
    _, _, _, _, info = env.step({0: 2, 1: 2, 2: 0, 3: 0})
    assert info["collision"] == 1


def test_single_sender_success_requires_packet_and_valid_target() -> None:
    env = WirelessMarkovGame(EnvParams(succ_idle=1.0, succ_cong=1.0))
    env.reset(seed=3)
    env.channel = 0
    env.buffers[:] = 0
    env.buffers[0] = 1
    _, _, _, _, info = env.step({0: 2, 1: 0, 2: 0, 3: 0})
    assert int(np.sum(info["success_vec"])) == 1


def test_invalid_target_triggers_penalty_signal() -> None:
    env = WirelessMarkovGame(EnvParams(topology="star"))
    env.reset(seed=5)
    env.buffers[:] = 1
    _, rewards, _, _, info = env.step({1: 3, 0: 0, 2: 0, 3: 0})
    assert info["invalid_vec"][1] == 1
    assert rewards[1] < 0.0

