from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


DEFAULT_TOPOLOGY = "all_to_all"
TOPOLOGY_LABELS = {
    "all_to_all": "全连接",
    "star": "星形",
    "ring": "环形",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def normalize_topology(topology: str) -> str:
    topology_name = str(topology).strip().lower()
    if topology_name not in TOPOLOGY_LABELS:
        raise ValueError(f"Unknown topology: {topology}")
    return topology_name


def topology_label(topology: str) -> str:
    return TOPOLOGY_LABELS[normalize_topology(topology)]


def topology_suffix(topology: str) -> str:
    topology_name = normalize_topology(topology)
    if topology_name == DEFAULT_TOPOLOGY:
        return ""
    return f"_{topology_name}"


def result_artifact_path(
    results_dir: str | Path,
    algo: str,
    artifact: str,
    topology: str = DEFAULT_TOPOLOGY,
) -> Path:
    algo_name = str(algo).lower()
    artifact_name = str(artifact).lower()
    suffix = topology_suffix(topology)
    base_dir = Path(results_dir)

    if artifact_name == "policy":
        ext = ".npz" if algo_name in {"iql", "value_iteration"} else ".pt"
        return base_dir / f"{algo_name}_policy{suffix}{ext}"

    if artifact_name in {"train_log", "action_hist", "summary"}:
        return base_dir / f"{algo_name}_{artifact_name}{suffix}.csv"

    raise ValueError(f"Unknown artifact: {artifact}")


def resolve_policy_artifact_path(
    results_dir: str | Path,
    algo: str,
    topology: str,
    allow_fallback: bool = False,
) -> tuple[Path, str]:
    requested_topology = normalize_topology(topology)
    requested_path = result_artifact_path(results_dir, algo, "policy", requested_topology)
    if requested_path.exists():
        return requested_path, requested_topology

    if allow_fallback and requested_topology != DEFAULT_TOPOLOGY:
        fallback_path = result_artifact_path(results_dir, algo, "policy", DEFAULT_TOPOLOGY)
        if fallback_path.exists():
            return fallback_path, DEFAULT_TOPOLOGY

    return requested_path, requested_topology


def topology_adjacency(n_agents: int, topology: str) -> np.ndarray:
    topology = normalize_topology(topology)
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
