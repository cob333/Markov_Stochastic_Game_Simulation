from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class QMIXConfig:
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 10000
    warmup_steps: int = 500
    train_every: int = 4
    tau: float = 0.01
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    agent_hidden: int = 64
    mixing_hidden: int = 32
    device: str = "cpu"

    @classmethod
    def from_config(cls, cfg: dict) -> "QMIXConfig":
        return cls(
            lr=float(cfg["lr"]),
            batch_size=int(cfg["batch_size"]),
            buffer_size=int(cfg["buffer_size"]),
            warmup_steps=int(cfg["warmup_steps"]),
            train_every=int(cfg["train_every"]),
            tau=float(cfg["tau"]),
            eps_start=float(cfg["eps_start"]),
            eps_end=float(cfg["eps_end"]),
            eps_decay=float(cfg["eps_decay"]),
            agent_hidden=int(cfg["agent_hidden"]),
            mixing_hidden=int(cfg["mixing_hidden"]),
            device=str(cfg["device"]),
        )


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.data: list[Any] = []
        self.pos = 0

    def add(self, item: Any) -> None:
        if len(self.data) < self.capacity:
            self.data.append(item)
            return
        self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Any]:
        indices = self.rng.choice(len(self.data), size=batch_size, replace=False)
        return [self.data[int(idx)] for idx in indices]

    def __len__(self) -> int:
        return len(self.data)


class AgentQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MixingNet(nn.Module):
    """QMIX mixing network with positive hypernetwork weights."""

    def __init__(self, n_agents: int, state_dim: int, mixing_hidden: int):
        super().__init__()
        self.n_agents = n_agents
        self.mixing_hidden = mixing_hidden
        hyper_hidden = max(32, 2 * mixing_hidden)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden),
        )
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )

    def forward(self, agent_q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = agent_q_values.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(
            batch_size, self.n_agents, self.mixing_hidden
        )
        b1 = self.hyper_b1(state).view(batch_size, 1, self.mixing_hidden)
        hidden = torch.bmm(agent_q_values.view(batch_size, 1, self.n_agents), w1) + b1
        hidden = torch.relu(hidden)

        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, self.mixing_hidden, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(batch_size, 1)


class QMIXTrainer:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        state_dim: int,
        cfg: QMIXConfig,
        seed: int,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(seed)
        self.replay = ReplayBuffer(cfg.buffer_size, seed=seed)

        self.agent_nets = nn.ModuleList(
            [AgentQNet(obs_dim, n_actions, cfg.agent_hidden) for _ in range(n_agents)]
        ).to(self.device)
        self.target_agent_nets = nn.ModuleList(
            [AgentQNet(obs_dim, n_actions, cfg.agent_hidden) for _ in range(n_agents)]
        ).to(self.device)
        self.mixer = MixingNet(n_agents, state_dim, cfg.mixing_hidden).to(self.device)
        self.target_mixer = MixingNet(n_agents, state_dim, cfg.mixing_hidden).to(self.device)
        self.optimizer = optim.Adam(
            list(self.agent_nets.parameters()) + list(self.mixer.parameters()),
            lr=cfg.lr,
        )
        self.hard_update()

    def hard_update(self) -> None:
        for idx in range(self.n_agents):
            self.target_agent_nets[idx].load_state_dict(self.agent_nets[idx].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_update(self) -> None:
        with torch.no_grad():
            for source, target in zip(self.agent_nets.parameters(), self.target_agent_nets.parameters()):
                target.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * source.data)
            for source, target in zip(self.mixer.parameters(), self.target_mixer.parameters()):
                target.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * source.data)

    def select_actions(
        self, obs_dict: dict[int, np.ndarray], eps: float, greedy: bool = False
    ) -> dict[int, int]:
        actions: dict[int, int] = {}
        for agent_id in range(self.n_agents):
            if (not greedy) and self.rng.random() < eps:
                actions[agent_id] = int(self.rng.integers(0, self.n_actions))
                continue
            obs_tensor = (
                torch.tensor(obs_dict[agent_id], dtype=torch.float32, device=self.device)
                .unsqueeze(0)
            )
            q_values = self.agent_nets[agent_id](obs_tensor).detach().cpu().numpy()[0]
            best_q = np.max(q_values)
            candidates = np.flatnonzero(np.isclose(q_values, best_q))
            actions[agent_id] = int(self.rng.choice(candidates))
        return actions

    def add_transition(self, transition: Any) -> None:
        self.replay.add(transition)

    def train_step(self, gamma: float) -> float | None:
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        states = torch.tensor(
            np.stack([item[0] for item in batch]), dtype=torch.float32, device=self.device
        )
        obs = torch.tensor(
            np.stack(
                [np.stack([item[1][agent_id] for agent_id in range(self.n_agents)]) for item in batch]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            np.stack(
                [
                    np.array([item[2][agent_id] for agent_id in range(self.n_agents)], dtype=np.int64)
                    for item in batch
                ]
            ),
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.tensor(
            np.array([item[3] for item in batch], dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.stack([item[4] for item in batch]), dtype=torch.float32, device=self.device
        )
        next_obs = torch.tensor(
            np.stack(
                [np.stack([item[5][agent_id] for agent_id in range(self.n_agents)]) for item in batch]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.tensor(
            np.array([item[6] for item in batch], dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )

        current_q_values = []
        next_q_values = []
        for agent_id in range(self.n_agents):
            q_all = self.agent_nets[agent_id](obs[:, agent_id, :])
            q_selected = q_all.gather(1, actions[:, agent_id].unsqueeze(1))
            current_q_values.append(q_selected)

            online_next = self.agent_nets[agent_id](next_obs[:, agent_id, :])
            next_action = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target_agent_nets[agent_id](next_obs[:, agent_id, :])
            target_selected = target_next.gather(1, next_action)
            next_q_values.append(target_selected)

        current_q_values = torch.cat(current_q_values, dim=1)
        next_q_values = torch.cat(next_q_values, dim=1)
        q_tot = self.mixer(current_q_values, states)
        with torch.no_grad():
            next_q_tot = self.target_mixer(next_q_values, next_states)
            target = rewards + gamma * (1.0 - done) * next_q_tot

        loss = torch.mean((q_tot - target) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_nets.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()
        self.soft_update()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "agent_nets": self.agent_nets.state_dict(),
                "mixer": self.mixer.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_nets.load_state_dict(checkpoint["agent_nets"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.hard_update()
