from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Advanced FL Assignment CLI")
    parser.add_argument(
        "--name",
        default="Advanced FL Assignment",
        help="Name to print in the greeting.",
    )
    subparsers = parser.add_subparsers(dest="command")

    fedavg_parser = subparsers.add_parser("fedavg", help="Run FedAvg experiment.")
    fedavg_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fedavg.yaml"),
        help="Path to FedAvg config file (JSON-compatible YAML).",
    )

    fedawa_parser = subparsers.add_parser("fedawa", help="Run FedAWA experiment.")
    fedawa_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fedawa.yaml"),
        help="Path to FedAWA config file (JSON-compatible YAML).",
    )

    dfl_parser = subparsers.add_parser("dfl", help="Run DFL baseline experiment.")
    dfl_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dfl.yaml"),
        help="Path to DFL config file (JSON-compatible YAML).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fedavg":
        from src.fl.fedavg import load_fedavg_config, run_fedavg_experiment

        config = load_fedavg_config(args.config)
        run_fedavg_experiment(config)
        return 0
    if args.command == "fedawa":
        from src.fl.fedawa import load_fedawa_config, run_fedawa_experiment

        config = load_fedawa_config(args.config)
        run_fedawa_experiment(config)
        return 0
    if args.command == "dfl":
        from src.dfl.simulator import load_dfl_config, run_dfl_experiment

        config = load_dfl_config(args.config)
        run_dfl_experiment(config)
        return 0

    print(f"Hello from {args.name}!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
