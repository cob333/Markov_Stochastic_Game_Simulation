from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

from wireless_marl.env import EnvParams
from wireless_marl.utils import clip01, topology_adjacency


@dataclass
class ValueIterationConfig:
    tol: float = 1.0e-6
    max_iters: int = 100

    @classmethod
    def from_config(cls, cfg: dict) -> "ValueIterationConfig":
        return cls(tol=float(cfg["tol"]), max_iters=int(cfg["max_iters"]))


def enumerate_joint_actions(n_agents: int, action_dim: int):
    return product(range(action_dim), repeat=n_agents)


def enumerate_states(n_agents: int) -> list[tuple[int, ...]]:
    states = []
    for channel in (0, 1):
        for buffers in product((0, 1), repeat=n_agents):
            states.append((channel, *buffers))
    return states


class ValueIterationPlanner:
    """Exact value iteration for the centralized team-reward MDP."""

    def __init__(self, params: EnvParams, cfg: ValueIterationConfig):
        self.params = params
        self.cfg = cfg
        self.n_agents = params.n_agents
        self.action_dim = 2 + (self.n_agents - 1)
        self.adjacency = topology_adjacency(self.n_agents, params.topology)
        self.states = enumerate_states(self.n_agents)
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.joint_actions = list(enumerate_joint_actions(self.n_agents, self.action_dim))
        self.values = np.zeros(len(self.states), dtype=np.float32)
        self.policy_indices = np.zeros(len(self.states), dtype=np.int64)

    def decode_target(self, agent_id: int, action: int) -> int | None:
        if action < 2:
            return None
        offset = action - 1
        target = (agent_id + offset) % self.n_agents
        if target == agent_id:
            return None
        return target

    def _arrival_distribution(
        self, buffers_after: tuple[int, ...]
    ) -> Iterable[tuple[tuple[int, ...], float]]:
        empty_positions = [idx for idx, value in enumerate(buffers_after) if value == 0]
        if not empty_positions:
            yield buffers_after, 1.0
            return

        for arrival_bits in product((0, 1), repeat=len(empty_positions)):
            prob = 1.0
            next_buffers = list(buffers_after)
            for position, bit in zip(empty_positions, arrival_bits):
                if bit == 1:
                    prob *= self.params.arrival_p
                    next_buffers[position] = 1
                else:
                    prob *= 1.0 - self.params.arrival_p
                    next_buffers[position] = 0
            yield tuple(next_buffers), prob

    def _expected_reward_and_transition(
        self, state: tuple[int, ...], joint_action: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
        channel = int(state[0])
        buffers = tuple(int(x) for x in state[1:])

        tx_vec = [0] * self.n_agents
        invalid_vec = [0] * self.n_agents
        for agent_id, action in enumerate(joint_action):
            if action >= 2:
                tx_vec[agent_id] = 1
                target = self.decode_target(agent_id, action)
                if target is None or self.adjacency[agent_id, target] == 0:
                    invalid_vec[agent_id] = 1

        k_tx = sum(tx_vec)
        collision = int(k_tx >= 2)
        success_agent = int(np.argmax(tx_vec)) if k_tx == 1 else None
        succ_prob = 0.0
        if success_agent is not None:
            if buffers[success_agent] == 1 and invalid_vec[success_agent] == 0:
                succ_prob = self.params.succ_cong if channel == 1 else self.params.succ_idle

        expected_team_reward = 0.0
        for agent_id, action in enumerate(joint_action):
            reward = 0.0
            if buffers[agent_id] == 1:
                reward -= self.params.c_delay

            if action == 0:
                reward -= self.params.e_wait
            elif action == 1:
                reward -= self.params.e_sleep
            else:
                reward -= self.params.e_tx
                if invalid_vec[agent_id] == 1:
                    reward -= self.params.c_invalid

                if collision and buffers[agent_id] == 1:
                    reward -= self.params.c_coll
                elif tx_vec[agent_id] == 1 and buffers[agent_id] == 1:
                    reward += succ_prob * self.params.r_succ
                    reward -= (1.0 - succ_prob) * self.params.c_fail

            expected_team_reward += reward

        p_congest_next = clip01(
            self.params.p_base + self.params.p_load * k_tx + self.params.p_persist * channel
        )

        transitions: dict[int, float] = {}
        success_cases: list[tuple[float, bool]]
        if success_agent is not None and succ_prob > 0.0:
            success_cases = [(succ_prob, True), (1.0 - succ_prob, False)]
        else:
            success_cases = [(1.0, False)]

        for success_case_prob, success_happened in success_cases:
            buffers_after = list(buffers)
            if success_happened and success_agent is not None:
                buffers_after[success_agent] = 0

            for next_buffers, buffer_prob in self._arrival_distribution(tuple(buffers_after)):
                for next_channel in (0, 1):
                    channel_prob = p_congest_next if next_channel == 1 else 1.0 - p_congest_next
                    prob = success_case_prob * buffer_prob * channel_prob
                    next_state = (next_channel, *next_buffers)
                    next_idx = self.state_to_idx[next_state]
                    transitions[next_idx] = transitions.get(next_idx, 0.0) + prob

        return expected_team_reward, transitions

    def run(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        values = self.values.copy()
        for iteration in range(1, self.cfg.max_iters + 1):
            new_values = np.zeros_like(values)
            delta = 0.0

            for state_idx, state in enumerate(self.states):
                best_q = float("-inf")
                best_action_idx = 0
                for action_idx, joint_action in enumerate(self.joint_actions):
                    reward, transitions = self._expected_reward_and_transition(state, joint_action)
                    q_value = reward + self.params.gamma * sum(
                        prob * float(values[next_idx])
                        for next_idx, prob in transitions.items()
                    )
                    if q_value > best_q:
                        best_q = q_value
                        best_action_idx = action_idx

                new_values[state_idx] = best_q
                self.policy_indices[state_idx] = best_action_idx
                delta = max(delta, abs(float(new_values[state_idx] - values[state_idx])))

            values = new_values
            history.append({"iteration": float(iteration), "delta": float(delta)})
            if delta < self.cfg.tol:
                break

        self.values = values
        return history

    def greedy_joint_action(self, state: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.joint_actions[int(self.policy_indices[self.state_to_idx[state]])])

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            values=self.values,
            policy_indices=self.policy_indices,
            states=np.array(self.states, dtype=np.int64),
            joint_actions=np.array(self.joint_actions, dtype=np.int64),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = np.load(path)
        self.values = checkpoint["values"].astype(np.float32)
        self.policy_indices = checkpoint["policy_indices"].astype(np.int64)
