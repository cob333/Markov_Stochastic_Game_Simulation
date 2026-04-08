from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from wireless_marl.algos.iql import IQLConfig, make_iql_agents, save_iql_agents
from wireless_marl.algos.value_iteration import ValueIterationConfig, ValueIterationPlanner
from wireless_marl.env import EnvParams, WirelessMarkovGame
from wireless_marl.utils import ensure_dir, set_global_seed


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_env(cfg: dict[str, Any]) -> WirelessMarkovGame:
    return WirelessMarkovGame(EnvParams.from_config(cfg))


def current_state_tuple(env: WirelessMarkovGame) -> tuple[int, ...]:
    return (int(env.channel), *[int(value) for value in env.buffers.tolist()])


def copy_obs_dict(obs_dict: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {agent_id: obs.copy() for agent_id, obs in obs_dict.items()}


def save_action_hist(path: str | Path, action_hist: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action", "count"])
        for action_id, count in enumerate(action_hist):
            writer.writerow([action_id, int(count)])


def save_summary(path: str | Path, metrics: dict[str, float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(metrics.keys()))
        writer.writerow([metrics[key] for key in metrics])


def evaluate_policy(
    env: WirelessMarkovGame,
    episodes: int,
    seed: int,
    action_selector: Callable[
        [dict[int, np.ndarray], np.ndarray, tuple[int, ...]], dict[int, int]
    ],
) -> dict[str, float | np.ndarray]:
    total_success = 0
    total_collisions = 0
    total_reward = 0.0
    total_energy = 0.0
    total_steps = 0
    action_hist = np.zeros(env.action_dim, dtype=np.int64)

    for episode in range(episodes):
        obs = env.reset(seed=seed + episode)
        done = False
        while not done:
            state_vec = env.get_state_vec().copy()
            state_tuple = current_state_tuple(env)
            actions = action_selector(obs, state_vec, state_tuple)
            for action in actions.values():
                action_hist[int(action)] += 1
            obs, rewards, terminated, truncated, info = env.step(actions)
            total_success += int(np.sum(info["success_vec"]))
            total_collisions += int(info["collision"])
            total_reward += float(sum(rewards.values()))
            total_energy += float(np.sum(info["energy_vec"]))
            total_steps += 1
            done = bool(terminated or truncated)

    per_step = max(1, total_steps)
    return {
        "throughput": total_success / per_step,
        "collision_rate": total_collisions / per_step,
        "avg_reward_per_agent": total_reward / (per_step * env.n_agents),
        "avg_energy_per_agent": total_energy / (per_step * env.n_agents),
        "action_hist": action_hist,
    }


def train_iql(cfg: dict[str, Any]) -> None:
    ensure_dir(RESULTS_DIR)
    set_global_seed(int(cfg["seed"]))

    env = make_env(cfg)
    iql_cfg = IQLConfig.from_config(cfg["iql"], gamma=float(cfg["gamma"]))

    train_cfg = cfg["train"]
    episodes = int(train_cfg["episodes"])
    eval_every = int(train_cfg["eval_every"])
    eval_episodes = int(train_cfg["eval_episodes"])
    agents = make_iql_agents(
        n_agents=env.n_agents,
        n_actions=env.action_dim,
        cfg=iql_cfg,
        seed=int(cfg["seed"]),
    )

    log_path = RESULTS_DIR / "iql_train_log.csv"
    action_hist_path = RESULTS_DIR / "iql_action_hist.csv"
    policy_path = RESULTS_DIR / "iql_policy.npz"

    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "eps",
                "throughput",
                "collision_rate",
                "avg_reward_per_agent",
                "avg_energy_per_agent",
            ]
        )

        for episode in range(1, episodes + 1):
            obs = env.reset(seed=int(cfg["seed"]) * 10000 + episode)
            done = False
            while not done:
                actions = {
                    agent_id: agents[agent_id].act(obs[agent_id], greedy=False)
                    for agent_id in range(env.n_agents)
                }
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                done = bool(terminated or truncated)
                for agent_id in range(env.n_agents):
                    agents[agent_id].update(
                        obs=obs[agent_id],
                        action=actions[agent_id],
                        reward=rewards[agent_id],
                        next_obs=next_obs[agent_id],
                        done=done,
                    )
                obs = next_obs

            for agent in agents:
                agent.decay_eps()

            if episode % eval_every == 0 or episode == episodes:
                metrics = evaluate_policy(
                    env=env,
                    episodes=eval_episodes,
                    seed=50000 + episode,
                    action_selector=lambda obs, _state_vec, _state_tuple: {
                        agent_id: agents[agent_id].act(obs[agent_id], greedy=True)
                        for agent_id in range(env.n_agents)
                    },
                )
                writer.writerow(
                    [
                        episode,
                        agents[0].eps,
                        metrics["throughput"],
                        metrics["collision_rate"],
                        metrics["avg_reward_per_agent"],
                        metrics["avg_energy_per_agent"],
                    ]
                )
                print(
                    "[IQL]",
                    f"episode={episode}",
                    f"throughput={metrics['throughput']:.4f}",
                    f"collision={metrics['collision_rate']:.4f}",
                    f"reward={metrics['avg_reward_per_agent']:.4f}",
                )

    final_metrics = evaluate_policy(
        env=env,
        episodes=eval_episodes,
        seed=90000,
        action_selector=lambda obs, _state_vec, _state_tuple: {
            agent_id: agents[agent_id].act(obs[agent_id], greedy=True)
            for agent_id in range(env.n_agents)
        },
    )
    save_action_hist(action_hist_path, final_metrics["action_hist"])
    save_iql_agents(policy_path, agents)
    print(f"Saved training log to {log_path}")
    print(f"Saved final action histogram to {action_hist_path}")
    print(f"Saved policy to {policy_path}")


def train_value_iteration(cfg: dict[str, Any]) -> None:
    ensure_dir(RESULTS_DIR)
    set_global_seed(int(cfg["seed"]))
    env = make_env(cfg)

    planner = ValueIterationPlanner(
        params=env.params,
        cfg=ValueIterationConfig.from_config(cfg["value_iteration"]),
    )
    history = planner.run()

    log_path = RESULTS_DIR / "value_iteration_train_log.csv"
    summary_path = RESULTS_DIR / "value_iteration_summary.csv"
    action_hist_path = RESULTS_DIR / "value_iteration_action_hist.csv"
    policy_path = RESULTS_DIR / "value_iteration_policy.npz"

    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "delta"])
        for item in history:
            writer.writerow([int(item["iteration"]), item["delta"]])

    eval_episodes = int(cfg["train"]["eval_episodes"])
    final_metrics = evaluate_policy(
        env=env,
        episodes=eval_episodes,
        seed=130000,
        action_selector=lambda _obs, _state_vec, state_tuple: {
            agent_id: int(action)
            for agent_id, action in enumerate(planner.greedy_joint_action(state_tuple))
        },
    )
    save_summary(
        summary_path,
        {
            "throughput": float(final_metrics["throughput"]),
            "collision_rate": float(final_metrics["collision_rate"]),
            "avg_reward_per_agent": float(final_metrics["avg_reward_per_agent"]),
            "avg_energy_per_agent": float(final_metrics["avg_energy_per_agent"]),
        },
    )
    save_action_hist(action_hist_path, final_metrics["action_hist"])
    planner.save(policy_path)
    print(
        "[VALUE_ITERATION]",
        f"iterations={len(history)}",
        f"final_delta={history[-1]['delta'] if history else 0.0:.6e}",
        f"throughput={final_metrics['throughput']:.4f}",
    )
    print(f"Saved planner log to {log_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved policy to {policy_path}")


def train_qmix(cfg: dict[str, Any]) -> None:
    from wireless_marl.algos.qmix import QMIXConfig, QMIXTrainer

    ensure_dir(RESULTS_DIR)
    set_global_seed(int(cfg["seed"]))
    env = make_env(cfg)
    qmix_cfg = QMIXConfig.from_config(cfg["qmix"])
    train_cfg = cfg["train"]
    episodes = int(train_cfg["episodes"])
    eval_every = int(train_cfg["eval_every"])
    eval_episodes = int(train_cfg["eval_episodes"])
    gamma = float(cfg["gamma"])

    trainer = QMIXTrainer(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        n_actions=env.action_dim,
        state_dim=env.state_dim,
        cfg=qmix_cfg,
        seed=int(cfg["seed"]),
    )

    eps = qmix_cfg.eps_start
    global_step = 0
    interval_losses: list[float] = []

    log_path = RESULTS_DIR / "qmix_train_log.csv"
    action_hist_path = RESULTS_DIR / "qmix_action_hist.csv"
    policy_path = RESULTS_DIR / "qmix_policy.pt"

    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "eps",
                "throughput",
                "collision_rate",
                "avg_reward_per_agent",
                "avg_energy_per_agent",
                "loss",
            ]
        )

        for episode in range(1, episodes + 1):
            obs = env.reset(seed=int(cfg["seed"]) * 10000 + episode)
            done = False
            while not done:
                state_vec = env.get_state_vec().copy()
                obs_snapshot = copy_obs_dict(obs)
                actions = trainer.select_actions(obs_snapshot, eps=eps, greedy=False)
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                done = bool(terminated or truncated)
                next_state_vec = env.get_state_vec().copy()
                next_obs_snapshot = copy_obs_dict(next_obs)
                team_reward = float(sum(rewards.values()))
                trainer.add_transition(
                    (
                        state_vec,
                        obs_snapshot,
                        actions.copy(),
                        team_reward,
                        next_state_vec,
                        next_obs_snapshot,
                        done,
                    )
                )
                global_step += 1
                if global_step % qmix_cfg.train_every == 0:
                    loss = trainer.train_step(gamma=gamma)
                    if loss is not None:
                        interval_losses.append(loss)
                obs = next_obs

            eps = max(qmix_cfg.eps_end, eps * qmix_cfg.eps_decay)

            if episode % eval_every == 0 or episode == episodes:
                metrics = evaluate_policy(
                    env=env,
                    episodes=eval_episodes,
                    seed=150000 + episode,
                    action_selector=lambda obs, _state_vec, _state_tuple: trainer.select_actions(
                        obs, eps=0.0, greedy=True
                    ),
                )
                avg_loss = float(np.mean(interval_losses)) if interval_losses else float("nan")
                writer.writerow(
                    [
                        episode,
                        eps,
                        metrics["throughput"],
                        metrics["collision_rate"],
                        metrics["avg_reward_per_agent"],
                        metrics["avg_energy_per_agent"],
                        avg_loss,
                    ]
                )
                interval_losses.clear()
                print(
                    "[QMIX]",
                    f"episode={episode}",
                    f"throughput={metrics['throughput']:.4f}",
                    f"collision={metrics['collision_rate']:.4f}",
                    f"loss={avg_loss:.4f}" if not np.isnan(avg_loss) else "loss=nan",
                )

    final_metrics = evaluate_policy(
        env=env,
        episodes=eval_episodes,
        seed=190000,
        action_selector=lambda obs, _state_vec, _state_tuple: trainer.select_actions(
            obs, eps=0.0, greedy=True
        ),
    )
    save_action_hist(action_hist_path, final_metrics["action_hist"])
    trainer.save(str(policy_path))
    print(f"Saved training log to {log_path}")
    print(f"Saved final action histogram to {action_hist_path}")
    print(f"Saved policy to {policy_path}")


def train_mappo(cfg: dict[str, Any], share_params: bool) -> None:
    from wireless_marl.algos.mappo import MAPPOConfig, MAPPOTrainer

    ensure_dir(RESULTS_DIR)
    set_global_seed(int(cfg["seed"]))
    env = make_env(cfg)
    mappo_cfg = MAPPOConfig.from_config(cfg["mappo"])
    mappo_cfg.share_params = share_params
    train_cfg = cfg["train"]
    episodes = int(train_cfg["episodes"])
    eval_every = int(train_cfg["eval_every"])
    eval_episodes = int(train_cfg["eval_episodes"])
    gamma = float(cfg["gamma"])
    algo_name = "mappo" if share_params else "ippo"

    trainer = MAPPOTrainer(
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        n_actions=env.action_dim,
        cfg=mappo_cfg,
        seed=int(cfg["seed"]),
    )

    log_path = RESULTS_DIR / f"{algo_name}_train_log.csv"
    action_hist_path = RESULTS_DIR / f"{algo_name}_action_hist.csv"
    policy_path = RESULTS_DIR / f"{algo_name}_policy.pt"

    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode",
                "throughput",
                "collision_rate",
                "avg_reward_per_agent",
                "avg_energy_per_agent",
                "policy_loss",
                "value_loss",
                "entropy",
            ]
        )

        for episode in range(1, episodes + 1):
            obs = env.reset(seed=int(cfg["seed"]) * 10000 + episode)
            rollout: dict[str, list[Any]] = {
                "states": [],
                "obs": [],
                "actions": [],
                "log_probs": [],
                "team_rewards": [],
                "dones": [],
                "values": [],
            }

            done = False
            while not done:
                state_vec = env.get_state_vec().copy()
                actions, log_probs, value = trainer.select_actions(
                    obs_dict=obs,
                    state_vec=state_vec,
                    greedy=False,
                )
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                done = bool(terminated or truncated)
                rollout["states"].append(state_vec)
                rollout["obs"].append(
                    np.stack([obs[agent_id] for agent_id in range(env.n_agents)])
                )
                rollout["actions"].append(
                    np.array(
                        [actions[agent_id] for agent_id in range(env.n_agents)],
                        dtype=np.int64,
                    )
                )
                rollout["log_probs"].append(log_probs.copy())
                rollout["team_rewards"].append(float(sum(rewards.values())))
                rollout["dones"].append(float(done))
                rollout["values"].append(float(value))
                obs = next_obs

            losses = trainer.update(rollout=rollout, gamma=gamma)

            if episode % eval_every == 0 or episode == episodes:
                metrics = evaluate_policy(
                    env=env,
                    episodes=eval_episodes,
                    seed=210000 + episode,
                    action_selector=lambda obs, state_vec, _state_tuple: trainer.select_actions(
                        obs_dict=obs,
                        state_vec=state_vec,
                        greedy=True,
                    )[0],
                )
                writer.writerow(
                    [
                        episode,
                        metrics["throughput"],
                        metrics["collision_rate"],
                        metrics["avg_reward_per_agent"],
                        metrics["avg_energy_per_agent"],
                        losses["policy_loss"],
                        losses["value_loss"],
                        losses["entropy"],
                    ]
                )
                print(
                    f"[{algo_name.upper()}]",
                    f"episode={episode}",
                    f"throughput={metrics['throughput']:.4f}",
                    f"collision={metrics['collision_rate']:.4f}",
                    f"policy_loss={losses['policy_loss']:.4f}",
                )

    final_metrics = evaluate_policy(
        env=env,
        episodes=eval_episodes,
        seed=250000,
        action_selector=lambda obs, state_vec, _state_tuple: trainer.select_actions(
            obs_dict=obs,
            state_vec=state_vec,
            greedy=True,
        )[0],
    )
    save_action_hist(action_hist_path, final_metrics["action_hist"])
    trainer.save(str(policy_path))
    print(f"Saved training log to {log_path}")
    print(f"Saved final action histogram to {action_hist_path}")
    print(f"Saved policy to {policy_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "config.yaml"),
    )
    parser.add_argument("--algo", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.algo:
        cfg["algo"] = args.algo

    algo = str(cfg["algo"]).lower()
    if algo == "iql":
        train_iql(cfg)
        return
    if algo == "value_iteration":
        train_value_iteration(cfg)
        return
    if algo == "qmix":
        train_qmix(cfg)
        return
    if algo == "mappo":
        train_mappo(cfg, share_params=True)
        return
    if algo == "ippo":
        train_mappo(cfg, share_params=False)
        return

    raise ValueError(f"Unsupported algorithm: {algo}")


if __name__ == "__main__":
    main()
