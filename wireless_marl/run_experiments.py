from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ALGOS = ["value_iteration", "iql", "qmix", "mappo"]


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(BASE_DIR / "config.yaml"))
    parser.add_argument("--topology", type=str, default="star")
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    for algo in args.algos:
        run_command(
            [
                sys.executable,
                "-u",
                str(BASE_DIR / "train.py"),
                "--config",
                args.config,
                "--algo",
                algo,
                "--topology",
                args.topology,
            ]
        )

        if args.eval:
            run_command(
                [
                    sys.executable,
                    "-u",
                    str(BASE_DIR / "eval.py"),
                    "--config",
                    args.config,
                    "--algo",
                    algo,
                    "--topology",
                    args.topology,
                    "--episodes",
                    str(args.eval_episodes),
                ]
            )

        if args.plot:
            run_command(
                [
                    sys.executable,
                    "-u",
                    str(BASE_DIR / "plot.py"),
                    "--algo",
                    algo,
                    "--topology",
                    args.topology,
                ]
            )


if __name__ == "__main__":
    main()
