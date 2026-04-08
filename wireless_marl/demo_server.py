from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import yaml
from flask import Flask, jsonify, request, send_from_directory

from wireless_marl.algos.iql import IQLConfig, load_iql_agents
from wireless_marl.algos.value_iteration import ValueIterationConfig, ValueIterationPlanner
from wireless_marl.env import EnvParams, WirelessMarkovGame
from wireless_marl.utils import (
    normalize_topology,
    resolve_policy_artifact_path,
    result_artifact_path,
    topology_label,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DEMO_DIR = BASE_DIR / "demo"

ALGO_LABELS = {
    "value_iteration": "值迭代",
    "iql": "IQL",
    "qmix": "QMIX",
    "mappo": "MAPPO",
    "ippo": "IPPO",
}

TORCH_ALGOS = {"qmix", "mappo", "ippo"}


@dataclass
class SessionState:
    session_id: str
    algo: str
    seed: int
    env: WirelessMarkovGame
    policy_path: str
    policy_topology: str
    actor: Callable[[dict[int, np.ndarray], np.ndarray, tuple[int, ...]], dict[int, int]]
    obs_dict: dict[int, np.ndarray]
    playback_rng: np.random.Generator
    history_window: int
    action_noise_eps: float
    total_steps: int = 0
    total_success: int = 0
    total_collisions: int = 0
    total_reward: float = 0.0
    total_energy: float = 0.0
    last_actions: dict[int, int] | None = None
    last_rewards: dict[int, float] | None = None
    last_info: dict[str, Any] | None = None
    terminated: bool = False
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "steps": [],
            "throughput": [],
            "collision_rate": [],
            "throughput_window": [],
            "collision_rate_window": [],
            "step_throughput": [],
            "step_collision": [],
            "avg_reward_per_agent": [],
            "avg_energy_per_agent": [],
        }
    )
    recent_events: list[str] = field(default_factory=list)


SESSIONS: dict[str, SessionState] = {}
SESSIONS_LOCK = Lock()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def current_state_tuple(env: WirelessMarkovGame) -> tuple[int, ...]:
    return (int(env.channel), *[int(value) for value in env.buffers.tolist()])


def action_label(env: WirelessMarkovGame, agent_id: int, action: int | None) -> str:
    if action is None:
        return "—"
    if action == 0:
        return "等待"
    if action == 1:
        return "待机"
    target = env.decode_target(agent_id, action)
    return f"发送至 {target}" if target is not None else "发送至 ?"


def apply_demo_action_noise(
    actions: dict[int, int],
    env: WirelessMarkovGame,
    rng: np.random.Generator,
    eps: float,
) -> dict[int, int]:
    if eps <= 0.0:
        return {agent_id: int(action) for agent_id, action in actions.items()}

    noisy_actions: dict[int, int] = {}
    for agent_id, action in actions.items():
        if rng.random() < eps:
            noisy_actions[agent_id] = int(rng.integers(0, env.action_dim))
        else:
            noisy_actions[agent_id] = int(action)
    return noisy_actions


def action_selector_for_algo(
    algo: str,
    cfg: dict[str, Any],
    env: WirelessMarkovGame,
) -> tuple[
    Callable[[dict[int, np.ndarray], np.ndarray, tuple[int, ...]], dict[int, int]],
    str,
    str,
]:
    policy_path, policy_topology = resolve_policy_artifact_path(
        RESULTS_DIR,
        algo,
        str(cfg["topology"]),
        allow_fallback=True,
    )
    if not policy_path.exists():
        raise FileNotFoundError(f"缺少已训练策略文件：{policy_path}")

    if algo == "iql":
        iql_cfg = IQLConfig.from_config(cfg["iql"], gamma=float(cfg["gamma"]))
        agents = load_iql_agents(
            path=policy_path,
            n_actions=env.action_dim,
            cfg=iql_cfg,
            seed=int(cfg["seed"]),
        )

        def act(
            obs_dict: dict[int, np.ndarray], _state_vec: np.ndarray, _state_tuple: tuple[int, ...]
        ) -> dict[int, int]:
            return {
                agent_id: agents[agent_id].act(obs_dict[agent_id], greedy=True)
                for agent_id in range(env.n_agents)
            }

        return act, str(policy_path), policy_topology

    if algo == "value_iteration":
        planner = ValueIterationPlanner(
            params=env.params,
            cfg=ValueIterationConfig.from_config(cfg["value_iteration"]),
        )
        planner.load(policy_path)

        def act(
            _obs_dict: dict[int, np.ndarray], _state_vec: np.ndarray, state_tuple: tuple[int, ...]
        ) -> dict[int, int]:
            joint_action = planner.greedy_joint_action(state_tuple)
            return {
                agent_id: int(action) for agent_id, action in enumerate(joint_action)
            }

        return act, str(policy_path), policy_topology

    if algo == "qmix":
        from wireless_marl.algos.qmix import QMIXConfig, QMIXTrainer

        trainer = QMIXTrainer(
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            n_actions=env.action_dim,
            state_dim=env.state_dim,
            cfg=QMIXConfig.from_config(cfg["qmix"]),
            seed=int(cfg["seed"]),
        )
        trainer.load(str(policy_path))

        def act(
            obs_dict: dict[int, np.ndarray], _state_vec: np.ndarray, _state_tuple: tuple[int, ...]
        ) -> dict[int, int]:
            return trainer.select_actions(obs_dict, eps=0.0, greedy=True)

        return act, str(policy_path), policy_topology

    if algo in {"mappo", "ippo"}:
        import torch

        from wireless_marl.algos.mappo import MAPPOConfig, MAPPOTrainer

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
        trainer.load(str(policy_path))
        # This task's PPO policies are mixed strategies. Deterministic argmax
        # playback collapses them into "always standby", so demo playback samples.
        torch.manual_seed(int(cfg["seed"]))

        def act(
            obs_dict: dict[int, np.ndarray], state_vec: np.ndarray, _state_tuple: tuple[int, ...]
        ) -> dict[int, int]:
            return trainer.select_actions(obs_dict=obs_dict, state_vec=state_vec, greedy=False)[0]

        return act, str(policy_path), policy_topology

    raise ValueError(f"不支持的算法：{algo}")


def available_models(topology: str) -> list[dict[str, Any]]:
    torch_installed = importlib.util.find_spec("torch") is not None
    topology_name = normalize_topology(topology)
    models = []
    for algo in ("value_iteration", "iql", "qmix", "mappo", "ippo"):
        path, policy_topology = resolve_policy_artifact_path(
            RESULTS_DIR,
            algo,
            topology_name,
            allow_fallback=True,
        )
        available = path.exists()
        reason = ""
        note = ""
        if not available:
            reason = "缺少策略文件"
        elif algo in TORCH_ALGOS and not torch_installed:
            available = False
            reason = "未安装 torch"
        elif policy_topology != topology_name:
            note = f"缺少{topology_label(topology_name)}专用策略，已回退到{topology_label(policy_topology)}策略"
        models.append(
            {
                "id": algo,
                "label": ALGO_LABELS[algo],
                "available": available,
                "reason": reason,
                "note": note,
                "path": str(path),
                "topology": topology_name,
                "topology_label": topology_label(topology_name),
                "policy_topology": policy_topology,
                "policy_topology_label": topology_label(policy_topology),
            }
        )
    return models


def push_event(session: SessionState, message: str) -> None:
    session.recent_events.append(message)
    session.recent_events = session.recent_events[-12:]


def metrics_payload(session: SessionState) -> dict[str, float]:
    if session.total_steps == 0:
        return {
            "throughput": 0.0,
            "collision_rate": 0.0,
            "avg_reward_per_agent": 0.0,
            "avg_energy_per_agent": 0.0,
        }

    return {
        "throughput": session.total_success / session.total_steps,
        "collision_rate": session.total_collisions / session.total_steps,
        "avg_reward_per_agent": session.total_reward
        / (session.total_steps * session.env.n_agents),
        "avg_energy_per_agent": session.total_energy
        / (session.total_steps * session.env.n_agents),
    }


def append_history(session: SessionState) -> None:
    metrics = metrics_payload(session)
    step_success = 0.0
    step_collision = 0.0
    if session.last_info is not None:
        step_success = float(np.sum(session.last_info["success_vec"]))
        step_collision = float(session.last_info["collision"])

    session.history["step_throughput"].append(step_success)
    session.history["step_collision"].append(step_collision)
    step_window = max(1, session.history_window)
    throughput_window = session.history["step_throughput"][-step_window:]
    collision_window = session.history["step_collision"][-step_window:]
    session.history["steps"].append(float(session.total_steps))
    session.history["throughput"].append(float(metrics["throughput"]))
    session.history["collision_rate"].append(float(metrics["collision_rate"]))
    session.history["throughput_window"].append(float(np.mean(throughput_window)))
    session.history["collision_rate_window"].append(float(np.mean(collision_window)))
    session.history["avg_reward_per_agent"].append(float(metrics["avg_reward_per_agent"]))
    session.history["avg_energy_per_agent"].append(float(metrics["avg_energy_per_agent"]))


def serialize_session(session: SessionState) -> dict[str, Any]:
    metrics = metrics_payload(session)
    last_info = session.last_info or {}
    last_actions = session.last_actions or {}
    last_rewards = session.last_rewards or {}

    agents_payload = []
    for agent_id in range(session.env.n_agents):
        obs = session.obs_dict[agent_id]
        agents_payload.append(
            {
                "id": agent_id,
                "buffer": int(session.env.buffers[agent_id]),
                "obs_channel": int(obs[0]),
                "obs_buffer": int(obs[1]),
                "last_action": last_actions.get(agent_id),
                "last_action_label": action_label(
                    session.env, agent_id, last_actions.get(agent_id)
                ),
                "last_reward": float(last_rewards.get(agent_id, 0.0)),
                "success": bool(
                    last_info.get("success_vec", np.zeros(session.env.n_agents))[agent_id]
                )
                if last_info
                else False,
                "invalid": bool(
                    last_info.get("invalid_vec", np.zeros(session.env.n_agents))[agent_id]
                )
                if last_info
                else False,
            }
        )

    last_step = None
    if session.last_info is not None and session.last_actions is not None:
        last_step = {
            "k_tx": int(session.last_info["k_tx"]),
            "collision": bool(session.last_info["collision"]),
            "p_congest_next": float(session.last_info["p_congest_next"]),
            "actions": {
                str(agent_id): action_label(session.env, agent_id, action)
                for agent_id, action in session.last_actions.items()
            },
            "team_reward": float(sum(session.last_rewards.values())) if session.last_rewards else 0.0,
        }

    return {
        "session_id": session.session_id,
        "algo": session.algo,
        "algo_label": ALGO_LABELS[session.algo],
        "policy_path": session.policy_path,
        "policy_topology": session.policy_topology,
        "policy_topology_label": topology_label(session.policy_topology),
        "seed": session.seed,
        "terminated": session.terminated,
        "step_index": session.total_steps,
        "channel": {
            "value": int(session.env.channel),
            "label": "空闲" if int(session.env.channel) == 0 else "拥堵",
        },
        "topology": {
            "type": session.env.params.topology,
            "label": topology_label(session.env.params.topology),
            "adjacency": session.env.adjacency.tolist(),
        },
        "buffers": [int(value) for value in session.env.buffers.tolist()],
        "metrics": metrics,
        "history_window": session.history_window,
        "agents": agents_payload,
        "last_step": last_step,
        "history": session.history,
        "recent_events": session.recent_events,
    }


def build_session(config_path: str | Path, algo: str, seed: int, topology: str) -> SessionState:
    cfg = load_config(config_path)
    cfg["algo"] = algo
    cfg["seed"] = seed
    cfg["topology"] = normalize_topology(topology)
    env = WirelessMarkovGame(EnvParams.from_config(cfg))
    obs_dict = env.reset(seed=seed)
    actor, policy_path, policy_topology = action_selector_for_algo(algo=algo, cfg=cfg, env=env)
    session = SessionState(
        session_id=uuid4().hex,
        algo=algo,
        seed=seed,
        env=env,
        policy_path=policy_path,
        policy_topology=policy_topology,
        actor=actor,
        obs_dict={agent_id: obs.copy() for agent_id, obs in obs_dict.items()},
        playback_rng=np.random.default_rng(seed + 2024),
        history_window=int(cfg.get("demo", {}).get("history_window", 20)),
        action_noise_eps=float(cfg.get("demo", {}).get("action_noise_eps", 0.0)),
    )
    event = (
        f"会话已重置，算法 {ALGO_LABELS[algo]}，拓扑 {topology_label(cfg['topology'])}，随机种子 {seed}。"
    )
    if policy_topology != cfg["topology"]:
        event += f" 当前使用{topology_label(policy_topology)}策略回放。"
    push_event(session, event)
    return session


def step_session(session: SessionState, manual_actions: list[int] | None = None) -> None:
    if session.terminated:
        raise RuntimeError("当前会话已经结束，请先重置。")

    state_vec = session.env.get_state_vec().copy()
    state_tuple = current_state_tuple(session.env)
    if manual_actions is not None:
        if len(manual_actions) != session.env.n_agents:
            raise ValueError("manual_actions 的长度必须等于设备数量")
        actions = {agent_id: int(manual_actions[agent_id]) for agent_id in range(session.env.n_agents)}
    else:
        actions = session.actor(session.obs_dict, state_vec, state_tuple)
        actions = apply_demo_action_noise(
            actions=actions,
            env=session.env,
            rng=session.playback_rng,
            eps=session.action_noise_eps,
        )

    next_obs, rewards, terminated, truncated, info = session.env.step(actions)
    session.obs_dict = {agent_id: obs.copy() for agent_id, obs in next_obs.items()}
    session.last_actions = actions.copy()
    session.last_rewards = rewards.copy()
    session.last_info = info
    session.total_steps += 1
    session.total_success += int(np.sum(info["success_vec"]))
    session.total_collisions += int(info["collision"])
    session.total_reward += float(sum(rewards.values()))
    session.total_energy += float(np.sum(info["energy_vec"]))
    session.terminated = bool(terminated or truncated)
    append_history(session)

    event_bits = [
        f"步数={session.total_steps}",
        f"信道={'空闲' if session.env.channel == 0 else '拥堵'}",
        f"发送设备数={int(info['k_tx'])}",
    ]
    if info["collision"]:
        event_bits.append("发生碰撞")
    if int(np.sum(info["success_vec"])) > 0:
        winners = [str(idx) for idx, val in enumerate(info["success_vec"]) if int(val) == 1]
        event_bits.append(f"成功设备={','.join(winners)}")
    push_event(session, " | ".join(event_bits))


def create_app(config_path: str | Path) -> Flask:
    app = Flask(__name__, static_folder=str(DEMO_DIR), static_url_path="/static")

    @app.get("/")
    def index():
        return send_from_directory(DEMO_DIR, "index.html")

    @app.get("/api/models")
    def models():
        topology = request.args.get("topology", "all_to_all")
        try:
            topology_name = normalize_topology(topology)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        return jsonify(
            {
                "topology": topology_name,
                "topology_label": topology_label(topology_name),
                "models": available_models(topology_name),
            }
        )

    @app.post("/api/session/reset")
    def reset():
        payload = request.get_json(silent=True) or {}
        algo = str(payload.get("algo", "iql")).lower()
        seed = int(payload.get("seed", 0))
        topology = str(payload.get("topology", "all_to_all"))
        try:
            session = build_session(
                config_path=config_path,
                algo=algo,
                seed=seed,
                topology=topology,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        with SESSIONS_LOCK:
            SESSIONS[session.session_id] = session
        return jsonify(serialize_session(session))

    @app.get("/api/session/<session_id>")
    def session_state(session_id: str):
        with SESSIONS_LOCK:
            session = SESSIONS.get(session_id)
        if session is None:
            return jsonify({"error": "未知会话"}), 404
        return jsonify(serialize_session(session))

    @app.post("/api/session/step")
    def step():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"error": "缺少 session_id"}), 400

        with SESSIONS_LOCK:
            session = SESSIONS.get(str(session_id))
        if session is None:
            return jsonify({"error": "未知会话"}), 404

        manual_actions = payload.get("manual_actions")
        try:
            step_session(session, manual_actions=manual_actions)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 400
        return jsonify(serialize_session(session))

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(BASE_DIR / "config.yaml"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
