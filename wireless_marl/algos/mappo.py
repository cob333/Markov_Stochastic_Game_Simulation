from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class MAPPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    batch_size: int = 512
    update_epochs: int = 4
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    hidden_size: int = 128
    share_params: bool = True
    device: str = "cpu"

    @classmethod
    def from_config(cls, cfg: dict) -> "MAPPOConfig":
        return cls(
            lr=float(cfg["lr"]),
            clip_eps=float(cfg["clip_eps"]),
            batch_size=int(cfg["batch_size"]),
            update_epochs=int(cfg["update_epochs"]),
            gae_lambda=float(cfg["gae_lambda"]),
            ent_coef=float(cfg["ent_coef"]),
            vf_coef=float(cfg["vf_coef"]),
            hidden_size=int(cfg["hidden_size"]),
            share_params=bool(cfg["share_params"]),
            device=str(cfg["device"]),
        )


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MAPPOTrainer:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        n_actions: int,
        cfg: MAPPOConfig,
        seed: int,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(seed)

        if cfg.share_params:
            self.shared_actor = ActorNet(obs_dim, n_actions, cfg.hidden_size).to(self.device)
            self.actors = None
            actor_params = list(self.shared_actor.parameters())
        else:
            self.shared_actor = None
            self.actors = nn.ModuleList(
                [ActorNet(obs_dim, n_actions, cfg.hidden_size) for _ in range(n_agents)]
            ).to(self.device)
            actor_params = list(self.actors.parameters())

        self.critic = CriticNet(state_dim, cfg.hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(actor_params, lr=cfg.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def _actor(self, agent_id: int) -> nn.Module:
        if self.shared_actor is not None:
            return self.shared_actor
        assert self.actors is not None
        return self.actors[agent_id]

    def select_actions(
        self, obs_dict: dict[int, np.ndarray], state_vec: np.ndarray, greedy: bool = False
    ) -> tuple[dict[int, int], np.ndarray, float]:
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        value = float(self.critic(state_tensor).item())
        actions: dict[int, int] = {}
        log_probs = np.zeros(self.n_agents, dtype=np.float32)

        for agent_id in range(self.n_agents):
            obs_tensor = (
                torch.tensor(obs_dict[agent_id], dtype=torch.float32, device=self.device)
                .unsqueeze(0)
            )
            logits = self._actor(agent_id)(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            if greedy:
                action_tensor = torch.argmax(logits, dim=1)
            else:
                action_tensor = dist.sample()
            actions[agent_id] = int(action_tensor.item())
            log_probs[agent_id] = float(dist.log_prob(action_tensor).item())

        return actions, log_probs, value

    def update(self, rollout: dict[str, list[Any]], gamma: float) -> dict[str, float]:
        states = np.stack(rollout["states"]).astype(np.float32)
        obs = np.stack(rollout["obs"]).astype(np.float32)
        actions = np.stack(rollout["actions"]).astype(np.int64)
        old_log_probs = np.stack(rollout["log_probs"]).astype(np.float32)
        rewards = np.array(rollout["team_rewards"], dtype=np.float32)
        dones = np.array(rollout["dones"], dtype=np.float32)
        values = np.array(rollout["values"] + [0.0], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        for idx in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[idx]
            delta = rewards[idx] + gamma * values[idx + 1] * non_terminal - values[idx]
            last_advantage = delta + gamma * self.cfg.gae_lambda * non_terminal * last_advantage
            advantages[idx] = last_advantage
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        return_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        flat_obs = obs.reshape(-1, self.obs_dim)
        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        repeated_advantages = np.repeat(advantages, self.n_agents)

        obs_tensor = torch.tensor(flat_obs, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(flat_actions, dtype=torch.int64, device=self.device)
        old_log_prob_tensor = torch.tensor(
            flat_old_log_probs, dtype=torch.float32, device=self.device
        )
        repeated_advantage_tensor = torch.tensor(
            repeated_advantages, dtype=torch.float32, device=self.device
        )

        actor_losses = []
        critic_losses = []
        entropies = []
        batch_size = max(1, min(self.cfg.batch_size, len(flat_obs)))
        indices = np.arange(len(flat_obs))

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                actor_idx = indices[start : start + batch_size]
                actor_obs = obs_tensor[actor_idx]
                actor_actions = action_tensor[actor_idx]
                actor_old_log_probs = old_log_prob_tensor[actor_idx]
                actor_advantages = repeated_advantage_tensor[actor_idx]

                if self.shared_actor is not None:
                    logits = self.shared_actor(actor_obs)
                else:
                    logits_parts = []
                    for flat_index in actor_idx:
                        agent_id = int(flat_index % self.n_agents)
                        logits_parts.append(
                            self._actor(agent_id)(
                                obs_tensor[flat_index].unsqueeze(0)
                            )
                        )
                    logits = torch.cat(logits_parts, dim=0)

                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actor_actions)
                ratio = torch.exp(new_log_probs - actor_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps
                )
                policy_loss = -torch.min(
                    ratio * actor_advantages, clipped_ratio * actor_advantages
                ).mean()
                entropy = dist.entropy().mean()

                self.actor_optimizer.zero_grad()
                (policy_loss - self.cfg.ent_coef * entropy).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.shared_actor.parameters()
                    if self.shared_actor is not None
                    else self.actors.parameters(),
                    10.0,
                )
                self.actor_optimizer.step()

                actor_losses.append(float(policy_loss.item()))
                entropies.append(float(entropy.item()))

            value_predictions = self.critic(state_tensor).squeeze(-1)
            critic_loss = torch.mean((value_predictions - return_tensor) ** 2)
            self.critic_optimizer.zero_grad()
            (self.cfg.vf_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()
            critic_losses.append(float(critic_loss.item()))

        return {
            "policy_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "value_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def save(self, path: str) -> None:
        payload = {"critic": self.critic.state_dict(), "cfg": self.cfg.__dict__}
        if self.shared_actor is not None:
            payload["shared_actor"] = self.shared_actor.state_dict()
        else:
            assert self.actors is not None
            payload["actors"] = self.actors.state_dict()
        torch.save(payload, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        if self.shared_actor is not None and "shared_actor" in checkpoint:
            self.shared_actor.load_state_dict(checkpoint["shared_actor"])
        elif self.actors is not None and "actors" in checkpoint:
            self.actors.load_state_dict(checkpoint["actors"])
