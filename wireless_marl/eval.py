from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path
from typing import Any

import yaml

from wireless_marl.algos.iql import IQLConfig, load_iql_agents
from wireless_marl.algos.value_iteration import ValueIterationConfig, ValueIterationPlanner
from wireless_marl.env import EnvParams, WirelessMarkovGame
from wireless_marl.train import RESULTS_DIR, evaluate_policy
from wireless_marl.utils import normalize_topology, result_artifact_path, set_global_seed, topology_label


BASE_DIR = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def print_metrics(metrics: dict[str, float | object]) -> None:
    print("Evaluation metrics:")
    print(f"  throughput: {float(metrics['throughput']):.4f}")
    print(f"  collision_rate: {float(metrics['collision_rate']):.4f}")
    print(f"  avg_reward_per_agent: {float(metrics['avg_reward_per_agent']):.4f}")
    print(f"  avg_energy_per_agent: {float(metrics['avg_energy_per_agent']):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "config.yaml"),
    )
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--topology", type=str, default=None)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.algo:
        cfg["algo"] = args.algo
    if args.topology:
        cfg["topology"] = normalize_topology(args.topology)

    algo = str(cfg["algo"]).lower()
    topology = str(cfg["topology"])
    set_global_seed(int(cfg["seed"]))
    env = WirelessMarkovGame(EnvParams.from_config(cfg))
    print(f"Topology: {topology_label(topology)}")

    if algo == "iql":
        policy_path = args.policy or str(result_artifact_path(RESULTS_DIR, "iql", "policy", topology))
        iql_cfg = IQLConfig.from_config(cfg["iql"], gamma=float(cfg["gamma"]))
        agents = load_iql_agents(
            path=policy_path,
            n_actions=env.action_dim,
            cfg=iql_cfg,
            seed=int(cfg["seed"]),
        )
        metrics = evaluate_policy(
            env=env,
            episodes=args.episodes,
            seed=123456,
            action_selector=lambda obs, _state_vec, _state_tuple: {
                agent_id: agents[agent_id].act(obs[agent_id], greedy=True)
                for agent_id in range(env.n_agents)
            },
        )
        print_metrics(metrics)
        return

    if algo == "qmix":
        from wireless_marl.algos.qmix import QMIXConfig, QMIXTrainer

        policy_path = args.policy or str(result_artifact_path(RESULTS_DIR, "qmix", "policy", topology))
        trainer = QMIXTrainer(
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            n_actions=env.action_dim,
            state_dim=env.state_dim,
            cfg=QMIXConfig.from_config(cfg["qmix"]),
            seed=int(cfg["seed"]),
        )
        trainer.load(policy_path)
        metrics = evaluate_policy(
            env=env,
            episodes=args.episodes,
            seed=123456,
            action_selector=lambda obs, _state_vec, _state_tuple: trainer.select_actions(
                obs, eps=0.0, greedy=True
            ),
        )
        print_metrics(metrics)
        return

    if algo in {"mappo", "ippo"}:
        import torch

        from wireless_marl.algos.mappo import MAPPOConfig, MAPPOTrainer

        policy_path = args.policy or str(
            result_artifact_path(RESULTS_DIR, algo, "policy", topology)
        )
        mappo_cfg = MAPPOConfig.from_config(cfg["mappo"])
        mappo_cfg.share_params = algo == "mappo"
        trainer = MAPPOTrainer(
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            n_actions=env.action_dim,
            cfg=mappo_cfg,
            seed=int(cfg["seed"]),
        )
        trainer.load(policy_path)
        # Default to stochastic evaluation for PPO-family policies in this task.
        torch.manual_seed(123456)
        metrics = evaluate_policy(
            env=env,
            episodes=args.episodes,
            seed=123456,
            action_selector=lambda obs, state_vec, _state_tuple: trainer.select_actions(
                obs_dict=obs,
                state_vec=state_vec,
                greedy=args.greedy,
            )[0],
        )
        print_metrics(metrics)
        return

    if algo == "value_iteration":
        policy_path = args.policy or str(
            result_artifact_path(RESULTS_DIR, "value_iteration", "policy", topology)
        )
        planner = ValueIterationPlanner(
            params=env.params,
            cfg=ValueIterationConfig.from_config(cfg["value_iteration"]),
        )
        planner.load(policy_path)
        metrics = evaluate_policy(
            env=env,
            episodes=args.episodes,
            seed=123456,
            action_selector=lambda _obs, _state_vec, state_tuple: {
                agent_id: int(action)
                for agent_id, action in enumerate(planner.greedy_joint_action(state_tuple))
            },
        )
        print_metrics(metrics)
        return

    raise ValueError(f"Unsupported algorithm: {algo}")


if __name__ == "__main__":
    main()
