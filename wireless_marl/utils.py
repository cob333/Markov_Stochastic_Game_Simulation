from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def topology_adjacency(n_agents: int, topology: str) -> np.ndarray:
    adjacency = np.zeros((n_agents, n_agents), dtype=np.int64)

    if topology == "all_to_all":
        adjacency[:] = 1
        np.fill_diagonal(adjacency, 0)
        return adjacency

    if topology == "star":
        for agent_id in range(1, n_agents):
            adjacency[0, agent_id] = 1
            adjacency[agent_id, 0] = 1
        return adjacency

    if topology == "ring":
        for agent_id in range(n_agents):
            adjacency[agent_id, (agent_id - 1) % n_agents] = 1
            adjacency[agent_id, (agent_id + 1) % n_agents] = 1
        np.fill_diagonal(adjacency, 0)
        return adjacency

    raise ValueError(f"Unknown topology: {topology}")

