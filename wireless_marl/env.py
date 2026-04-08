from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wireless_marl.utils import clip01, topology_adjacency


@dataclass
class EnvParams:
    n_agents: int = 4
    episode_len: int = 200
    gamma: float = 0.95
    arrival_p: float = 0.3
    succ_idle: float = 0.95
    succ_cong: float = 0.60
    p_base: float = 0.05
    p_load: float = 0.20
    p_persist: float = 0.50
    r_succ: float = 1.0
    c_coll: float = 1.0
    c_fail: float = 0.3
    c_delay: float = 0.01
    c_invalid: float = 0.2
    e_tx: float = 0.05
    e_wait: float = 0.005
    e_sleep: float = 0.001
    p_obs_correct: float = 0.9
    topology: str = "all_to_all"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EnvParams":
        return cls(
            n_agents=int(cfg["n_agents"]),
            episode_len=int(cfg["episode_len"]),
            gamma=float(cfg["gamma"]),
            arrival_p=float(cfg["arrival_p"]),
            succ_idle=float(cfg["succ_idle"]),
            succ_cong=float(cfg["succ_cong"]),
            p_base=float(cfg["p_base"]),
            p_load=float(cfg["p_load"]),
            p_persist=float(cfg["p_persist"]),
            r_succ=float(cfg["r_succ"]),
            c_coll=float(cfg["c_coll"]),
            c_fail=float(cfg["c_fail"]),
            c_delay=float(cfg["c_delay"]),
            c_invalid=float(cfg["c_invalid"]),
            e_tx=float(cfg["e_tx"]),
            e_wait=float(cfg["e_wait"]),
            e_sleep=float(cfg["e_sleep"]),
            p_obs_correct=float(cfg["p_obs_correct"]),
            topology=str(cfg["topology"]),
        )


class WirelessMarkovGame:
    """Two-state channel plus binary buffers for four competing devices."""

    def __init__(self, params: EnvParams):
        self.params = params
        self.n_agents = params.n_agents
        self.obs_dim = 2
        self.state_dim = self.n_agents + 1
        self.action_dim = 2 + (self.n_agents - 1)
        self.adjacency = topology_adjacency(self.n_agents, params.topology)
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.channel = 0
        self.buffers = np.zeros(self.n_agents, dtype=np.int64)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> dict[int, np.ndarray]:
        if seed is not None:
            self.seed(seed)
        self.t = 0
        self.channel = int(self.rng.integers(0, 2))
        self.buffers = self.rng.binomial(1, 0.5, size=self.n_agents).astype(np.int64)
        return self._obs()

    def _obs(self) -> dict[int, np.ndarray]:
        # The appendix design uses one shared noisy channel sample per step.
        noisy_channel = (
            self.channel
            if self.rng.random() < self.params.p_obs_correct
            else 1 - self.channel
        )
        return {
            agent_id: np.array([noisy_channel, self.buffers[agent_id]], dtype=np.int64)
            for agent_id in range(self.n_agents)
        }

    def decode_target(self, agent_id: int, action: int) -> int | None:
        if action < 2:
            return None
        offset = action - 1
        target = (agent_id + offset) % self.n_agents
        if target == agent_id:
            return None
        return target

    def get_state_vec(self) -> np.ndarray:
        return np.concatenate(
            [np.array([self.channel], dtype=np.float32), self.buffers.astype(np.float32)]
        )

    def step(
        self, actions: dict[int, int]
    ) -> tuple[dict[int, np.ndarray], dict[int, float], bool, bool, dict[str, Any]]:
        tx_vec = np.zeros(self.n_agents, dtype=np.int64)
        invalid_vec = np.zeros(self.n_agents, dtype=np.int64)

        for agent_id in range(self.n_agents):
            action = int(actions.get(agent_id, 0))
            if action >= 2:
                tx_vec[agent_id] = 1
                target = self.decode_target(agent_id, action)
                if target is None or self.adjacency[agent_id, target] == 0:
                    invalid_vec[agent_id] = 1

        k_tx = int(tx_vec.sum())
        collision = int(k_tx >= 2)
        success_vec = np.zeros(self.n_agents, dtype=np.int64)
        succ_prob = (
            self.params.succ_cong if self.channel == 1 else self.params.succ_idle
        )

        if k_tx == 1:
            agent_id = int(np.argmax(tx_vec))
            if self.buffers[agent_id] == 1 and invalid_vec[agent_id] == 0:
                if self.rng.random() < succ_prob:
                    success_vec[agent_id] = 1

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        energy_vec = np.zeros(self.n_agents, dtype=np.float32)

        for agent_id in range(self.n_agents):
            action = int(actions.get(agent_id, 0))

            if self.buffers[agent_id] == 1:
                rewards[agent_id] -= self.params.c_delay

            if action == 0:
                rewards[agent_id] -= self.params.e_wait
                energy_vec[agent_id] = self.params.e_wait
            elif action == 1:
                rewards[agent_id] -= self.params.e_sleep
                energy_vec[agent_id] = self.params.e_sleep
            else:
                rewards[agent_id] -= self.params.e_tx
                energy_vec[agent_id] = self.params.e_tx

                if invalid_vec[agent_id] == 1:
                    rewards[agent_id] -= self.params.c_invalid

                if collision and self.buffers[agent_id] == 1:
                    rewards[agent_id] -= self.params.c_coll
                elif tx_vec[agent_id] == 1:
                    if success_vec[agent_id] == 1:
                        rewards[agent_id] += self.params.r_succ
                    elif self.buffers[agent_id] == 1:
                        rewards[agent_id] -= self.params.c_fail

        for agent_id in range(self.n_agents):
            if success_vec[agent_id] == 1:
                self.buffers[agent_id] = 0

        arrivals = self.rng.binomial(1, self.params.arrival_p, size=self.n_agents).astype(
            np.int64
        )
        for agent_id in range(self.n_agents):
            if self.buffers[agent_id] == 0:
                self.buffers[agent_id] = arrivals[agent_id]

        p_congest_next = clip01(
            self.params.p_base
            + self.params.p_load * k_tx
            + self.params.p_persist * self.channel
        )
        self.channel = int(self.rng.random() < p_congest_next)

        self.t += 1
        terminated = self.t >= self.params.episode_len
        truncated = False

        info = {
            "t": self.t,
            "k_tx": k_tx,
            "collision": collision,
            "success_vec": success_vec.copy(),
            "invalid_vec": invalid_vec.copy(),
            "energy_vec": energy_vec.copy(),
            "buffers": self.buffers.copy(),
            "channel": self.channel,
            "p_congest_next": p_congest_next,
        }
        reward_dict = {
            agent_id: float(rewards[agent_id]) for agent_id in range(self.n_agents)
        }
        return self._obs(), reward_dict, terminated, truncated, info

