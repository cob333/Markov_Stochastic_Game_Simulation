from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from wireless_marl.train import RESULTS_DIR
from wireless_marl.utils import (
    ensure_dir,
    normalize_topology,
    result_artifact_path,
    topology_suffix,
)


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "outputs" / "figs"
PLOT_TOPOLOGY_LABELS = {
    "all_to_all": "All-to-All",
    "star": "Star",
    "ring": "Ring",
}


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_topology_label(topology: str) -> str:
    return PLOT_TOPOLOGY_LABELS.get(normalize_topology(topology), normalize_topology(topology))


def plot_metric_curves(log_rows: list[dict[str, str]], algo: str, topology: str) -> None:
    episodes = [int(row["episode"]) for row in log_rows]
    throughput = [float(row["throughput"]) for row in log_rows]
    collision = [float(row["collision_rate"]) for row in log_rows]
    reward = [float(row["avg_reward_per_agent"]) for row in log_rows]
    topo_label = plot_topology_label(topology)
    suffix = topology_suffix(topology)

    plt.figure()
    plt.plot(episodes, throughput, label="throughput")
    plt.plot(episodes, collision, label="collision_rate")
    plt.xlabel("episode")
    plt.ylabel("metric")
    plt.title(f"{algo.upper()} Training Metrics ({topo_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{algo}_training_metrics{suffix}.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(episodes, reward, label="avg_reward_per_agent")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title(f"{algo.upper()} Reward Curve ({topo_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{algo}_reward_curve{suffix}.png", dpi=180)
    plt.close()

    last_row = log_rows[-1]
    plt.figure()
    plt.bar(
        ["throughput", "collision", "reward"],
        [
            float(last_row["throughput"]),
            float(last_row["collision_rate"]),
            float(last_row["avg_reward_per_agent"]),
        ],
    )
    plt.title(f"{algo.upper()} Final Evaluation ({topo_label})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{algo}_final_bar{suffix}.png", dpi=180)
    plt.close()


def plot_action_pie(hist_rows: list[dict[str, str]], algo: str, topology: str) -> None:
    labels = [f"a{row['action']}" for row in hist_rows]
    counts = [int(row["count"]) for row in hist_rows]
    topo_label = plot_topology_label(topology)
    suffix = topology_suffix(topology)
    plt.figure()
    plt.pie(counts, labels=labels, autopct="%1.1f%%")
    plt.title(f"{algo.upper()} Action Distribution ({topo_label})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{algo}_action_pie{suffix}.png", dpi=180)
    plt.close()


def plot_value_iteration(
    log_rows: list[dict[str, str]], summary_rows: list[dict[str, str]], topology: str
) -> None:
    iterations = [int(row["iteration"]) for row in log_rows]
    deltas = [float(row["delta"]) for row in log_rows]
    topo_label = plot_topology_label(topology)
    suffix = topology_suffix(topology)

    plt.figure()
    plt.plot(iterations, deltas, label="delta")
    plt.xlabel("iteration")
    plt.ylabel("bellman residual")
    plt.title(f"Value Iteration Convergence ({topo_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"value_iteration_delta_curve{suffix}.png", dpi=180)
    plt.close()

    summary = summary_rows[0]
    plt.figure()
    plt.bar(
        ["throughput", "collision", "reward"],
        [
            float(summary["throughput"]),
            float(summary["collision_rate"]),
            float(summary["avg_reward_per_agent"]),
        ],
    )
    plt.title(f"Value Iteration Final Evaluation ({topo_label})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"value_iteration_final_bar{suffix}.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="iql")
    parser.add_argument("--topology", type=str, default=None)
    args = parser.parse_args()
    ensure_dir(FIG_DIR)
    algo = str(args.algo).lower()
    topology = normalize_topology(args.topology or "all_to_all")

    if algo == "value_iteration":
        log_path = result_artifact_path(RESULTS_DIR, "value_iteration", "train_log", topology)
        hist_path = result_artifact_path(RESULTS_DIR, "value_iteration", "action_hist", topology)
        summary_path = result_artifact_path(RESULTS_DIR, "value_iteration", "summary", topology)
        if not log_path.exists():
            raise FileNotFoundError(f"Missing training log: {log_path}")
        if not hist_path.exists():
            raise FileNotFoundError(f"Missing action histogram: {hist_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")

        log_rows = read_csv_dicts(log_path)
        hist_rows = read_csv_dicts(hist_path)
        summary_rows = read_csv_dicts(summary_path)
        if not log_rows or not hist_rows or not summary_rows:
            raise RuntimeError("Value iteration result files are empty.")

        plot_value_iteration(log_rows, summary_rows, topology=topology)
        plot_action_pie(hist_rows, algo=algo, topology=topology)
        print(f"Saved figures to {FIG_DIR}")
        return

    log_path = result_artifact_path(RESULTS_DIR, algo, "train_log", topology)
    hist_path = result_artifact_path(RESULTS_DIR, algo, "action_hist", topology)
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log: {log_path}")
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing action histogram: {hist_path}")

    log_rows = read_csv_dicts(log_path)
    hist_rows = read_csv_dicts(hist_path)
    if not log_rows:
        raise RuntimeError("Training log is empty.")
    if not hist_rows:
        raise RuntimeError("Action histogram is empty.")

    plot_metric_curves(log_rows, algo=algo, topology=topology)
    plot_action_pie(hist_rows, algo=algo, topology=topology)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
