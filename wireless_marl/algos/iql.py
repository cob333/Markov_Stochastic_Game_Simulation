from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IQLConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995

    @classmethod
    def from_config(cls, cfg: dict, gamma: float) -> "IQLConfig":
        return cls(
            alpha=float(cfg["alpha"]),
            gamma=float(gamma),
            eps_start=float(cfg["eps_start"]),
            eps_end=float(cfg["eps_end"]),
            eps_decay=float(cfg["eps_decay"]),
        )


class IQLAgent:
    """Tabular IQL over local observations (channel_obs, local_buffer)."""

    def __init__(self, n_actions: int, cfg: IQLConfig, seed: int = 0):
        self.n_actions = n_actions
        self.cfg = cfg
        self.eps = cfg.eps_start
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((4, n_actions), dtype=np.float32)

    @staticmethod
    def obs_to_idx(obs: np.ndarray) -> int:
        channel_obs = int(obs[0])
        local_buffer = int(obs[1])
        return channel_obs * 2 + local_buffer

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        state_idx = self.obs_to_idx(obs)
        if (not greedy) and self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.n_actions))
        q_values = self.q_table[state_idx]
        best_value = q_values.max()
        candidates = np.flatnonzero(np.isclose(q_values, best_value))
        return int(self.rng.choice(candidates))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        state_idx = self.obs_to_idx(obs)
        next_state_idx = self.obs_to_idx(next_obs)
        td_target = reward
        if not done:
            td_target += self.cfg.gamma * float(self.q_table[next_state_idx].max())
        current_value = self.q_table[state_idx, action]
        self.q_table[state_idx, action] = (
            (1.0 - self.cfg.alpha) * current_value + self.cfg.alpha * td_target
        )

    def decay_eps(self) -> None:
        self.eps = max(self.cfg.eps_end, self.eps * self.cfg.eps_decay)


def make_iql_agents(
    n_agents: int, n_actions: int, cfg: IQLConfig, seed: int
) -> list[IQLAgent]:
    return [
        IQLAgent(n_actions=n_actions, cfg=cfg, seed=seed + 97 * agent_id)
        for agent_id in range(n_agents)
    ]


def save_iql_agents(path: str | Path, agents: list[IQLAgent]) -> None:
    payload = {"n_agents": len(agents)}
    for agent_id, agent in enumerate(agents):
        payload[f"q_{agent_id}"] = agent.q_table
        payload[f"eps_{agent_id}"] = np.array([agent.eps], dtype=np.float32)
    np.savez(path, **payload)


def load_iql_agents(
    path: str | Path, n_actions: int, cfg: IQLConfig, seed: int = 0
) -> list[IQLAgent]:
    data = np.load(path)
    n_agents = int(data["n_agents"])
    agents = make_iql_agents(n_agents=n_agents, n_actions=n_actions, cfg=cfg, seed=seed)
    for agent_id, agent in enumerate(agents):
        agent.q_table = data[f"q_{agent_id}"].astype(np.float32)
        agent.eps = float(data[f"eps_{agent_id}"][0])
    return agents

