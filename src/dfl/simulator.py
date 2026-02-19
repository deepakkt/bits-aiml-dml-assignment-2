from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.fl.fedavg import weighted_average_state_dicts
from src.plotting.line_plot import save_accuracy_line_plot

COMMUNICATION_ROUND_DEFINITION = (
    "One communication round includes (1) local training on each agent's private data, "
    "followed by (2) random pairwise model encounters where each encountered pair replaces "
    "both local models with their element-wise parameter average."
)


@dataclass(frozen=True)
class DFLConfig:
    experiment_name: str
    seed: int
    device: str
    extracted_dir: Path
    index_path: Path
    train_test_manifest_path: Path
    partition_manifest_path: Path
    test_split: str
    image_size: tuple[int, int]
    num_agents: int
    dirichlet_alpha: float
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    encounters_per_round: int
    num_classes: int
    input_channels: int
    metrics_path: Path
    models_dir: Path
    plot_path: Path
    clean_artifacts: bool = True

    def to_serializable_dict(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "extracted_dir": str(self.extracted_dir),
            "index_path": str(self.index_path),
            "train_test_manifest_path": str(self.train_test_manifest_path),
            "partition_manifest_path": str(self.partition_manifest_path),
            "test_split": self.test_split,
            "image_size": [self.image_size[0], self.image_size[1]],
            "num_agents": self.num_agents,
            "dirichlet_alpha": self.dirichlet_alpha,
            "rounds": self.rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "encounters_per_round": self.encounters_per_round,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "metrics_path": str(self.metrics_path),
            "models_dir": str(self.models_dir),
            "plot_path": str(self.plot_path),
            "communication_round_definition": COMMUNICATION_ROUND_DEFINITION,
            "clean_artifacts": self.clean_artifacts,
        }


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for '{key}' in DFL config.")
    return value


def _validate_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _validate_positive_float(name: str, value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _parse_image_size(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or len(raw) != 2:
        raise ValueError("dataset.image_size must be a 2-item sequence, e.g. [64, 64].")
    width = _validate_positive_int("dataset.image_size[0]", int(raw[0]))
    height = _validate_positive_int("dataset.image_size[1]", int(raw[1]))
    return (width, height)


def load_dfl_config(config_path: Path) -> DFLConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        try:
            payload = json.load(config_file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "DFL config must be valid JSON (JSON is valid YAML). "
                f"Could not parse {config_path}."
            ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("DFL config root must be a mapping object.")

    dataset_cfg = _require_mapping(payload, "dataset")
    training_cfg = _require_mapping(payload, "training")
    model_cfg = _require_mapping(payload, "model")
    dfl_cfg = _require_mapping(payload, "dfl")
    artifacts_cfg = _require_mapping(payload, "artifacts")

    num_agents = _validate_positive_int("training.num_agents", int(training_cfg["num_agents"]))
    encounters_per_round = _validate_positive_int(
        "dfl.encounters_per_round",
        int(dfl_cfg.get("encounters_per_round", num_agents // 2)),
    )
    max_pairs = num_agents // 2
    if encounters_per_round > max_pairs:
        raise ValueError(
            "dfl.encounters_per_round must be <= floor(num_agents / 2). "
            f"Got encounters_per_round={encounters_per_round}, max={max_pairs}."
        )

    return DFLConfig(
        experiment_name=str(payload.get("experiment_name", "dfl")),
        seed=int(payload.get("seed", 42)),
        device=str(payload.get("device", "cpu")),
        extracted_dir=Path(str(dataset_cfg["extracted_dir"])),
        index_path=Path(str(dataset_cfg["index_path"])),
        train_test_manifest_path=Path(str(dataset_cfg["train_test_manifest_path"])),
        partition_manifest_path=Path(str(dataset_cfg["partition_manifest_path"])),
        test_split=str(dataset_cfg.get("test_split", "test")),
        image_size=_parse_image_size(dataset_cfg.get("image_size", [64, 64])),
        num_agents=num_agents,
        dirichlet_alpha=_validate_positive_float(
            "training.dirichlet_alpha",
            float(training_cfg.get("dirichlet_alpha", 0.5)),
        ),
        rounds=_validate_positive_int("training.rounds", int(training_cfg["rounds"])),
        local_epochs=_validate_positive_int(
            "training.local_epochs",
            int(training_cfg["local_epochs"]),
        ),
        batch_size=_validate_positive_int("training.batch_size", int(training_cfg["batch_size"])),
        learning_rate=_validate_positive_float(
            "training.learning_rate",
            float(training_cfg["learning_rate"]),
        ),
        momentum=float(training_cfg.get("momentum", 0.0)),
        encounters_per_round=encounters_per_round,
        num_classes=_validate_positive_int("model.num_classes", int(model_cfg["num_classes"])),
        input_channels=_validate_positive_int(
            "model.input_channels",
            int(model_cfg.get("input_channels", 3)),
        ),
        metrics_path=Path(str(artifacts_cfg["metrics_path"])),
        models_dir=Path(str(artifacts_cfg["models_dir"])),
        plot_path=Path(str(artifacts_cfg["plot_path"])),
        clean_artifacts=bool(payload.get("clean_artifacts", True)),
    )


def set_global_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested_device: str) -> Any:
    import torch

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested_device}' but CUDA is unavailable in this environment."
        )
    return torch.device(requested_device)


def _prepare_artifact_paths(config: DFLConfig) -> None:
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.plot_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.clean_artifacts:
        return

    if config.metrics_path.exists():
        config.metrics_path.unlink()
    if config.plot_path.exists():
        config.plot_path.unlink()
    for snapshot_path in config.models_dir.glob("round_*.pkl"):
        snapshot_path.unlink()


def _extract_agent_ids(config: DFLConfig) -> list[str]:
    from src.data.dataset import read_split_manifest

    partition_manifest = read_split_manifest(config.partition_manifest_path)
    partition_num_clients = int(partition_manifest.get("num_clients", -1))
    if partition_num_clients != config.num_agents:
        raise ValueError(
            "Agent count mismatch between config and partition manifest: "
            f"{config.num_agents=} vs {partition_num_clients=}."
        )

    partition_params = partition_manifest.get("partition_params", {})
    alpha_value = float(partition_params.get("alpha", float("nan")))
    if not math.isclose(alpha_value, config.dirichlet_alpha, abs_tol=1e-12):
        raise ValueError(
            "Dirichlet alpha mismatch between config and partition manifest: "
            f"{config.dirichlet_alpha=} vs manifest_alpha={alpha_value}."
        )

    clients = partition_manifest.get("clients", {})
    if not isinstance(clients, Mapping) or not clients:
        raise ValueError("Partition manifest contains no clients.")

    agent_ids = sorted(str(client_id) for client_id in clients)
    if len(agent_ids) != config.num_agents:
        raise ValueError(f"Expected {config.num_agents} agents but found {len(agent_ids)}.")
    return agent_ids


def sample_random_pairings(
    agent_ids: Sequence[str],
    rng: random.Random,
    *,
    max_pairs: int | None = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    if not agent_ids:
        raise ValueError("agent_ids must not be empty.")

    normalized_agent_ids = [str(agent_id) for agent_id in agent_ids]
    if len(set(normalized_agent_ids)) != len(normalized_agent_ids):
        raise ValueError("agent_ids must be unique for pairwise sampling.")

    shuffled_ids = list(normalized_agent_ids)
    rng.shuffle(shuffled_ids)

    max_possible_pairs = len(shuffled_ids) // 2
    if max_pairs is None:
        selected_pairs = max_possible_pairs
    else:
        if max_pairs <= 0:
            raise ValueError(f"max_pairs must be > 0, got {max_pairs}")
        if max_pairs > max_possible_pairs:
            raise ValueError(
                f"max_pairs must be <= {max_possible_pairs} for {len(shuffled_ids)} agents."
            )
        selected_pairs = max_pairs

    paired_count = selected_pairs * 2
    pairs = [(shuffled_ids[index], shuffled_ids[index + 1]) for index in range(0, paired_count, 2)]
    unpaired_agents = shuffled_ids[paired_count:]
    return pairs, unpaired_agents


def _build_round_metrics_row(
    *,
    round_index: int,
    mean_train_loss: float,
    mean_train_accuracy: float,
    mean_test_loss: float,
    mean_test_accuracy: float,
    total_train_examples: int,
    num_agents: int,
    num_encounters: int,
    unpaired_agents: int,
) -> dict[str, float | int]:
    return {
        "round": round_index,
        "mean_train_loss": mean_train_loss,
        "mean_train_accuracy": mean_train_accuracy,
        "mean_test_loss": mean_test_loss,
        "mean_test_accuracy": mean_test_accuracy,
        "total_train_examples": total_train_examples,
        "num_agents": num_agents,
        "num_encounters": num_encounters,
        "unpaired_agents": unpaired_agents,
    }


def run_dfl_experiment(config: DFLConfig) -> list[dict[str, float | int]]:
    import torch
    from torch.utils.data import DataLoader

    from src.common.metrics import MetricsCSVLogger
    from src.common.serialization import save_model_snapshot
    from src.common.train_eval import evaluate_model, train_local_epochs
    from src.data.torch_dataset import (
        build_image_dataset_from_client,
        build_image_dataset_from_split,
    )
    from src.models.simple_cnn import SimpleCNN

    set_global_seed(config.seed)
    rng = random.Random(config.seed)
    device = _resolve_device(config.device)
    _prepare_artifact_paths(config)

    agent_ids = _extract_agent_ids(config)
    agent_datasets = {
        agent_id: build_image_dataset_from_client(
            extracted_dir=config.extracted_dir,
            index_path=config.index_path,
            partition_manifest_path=config.partition_manifest_path,
            client_id=agent_id,
            image_size=config.image_size,
        )
        for agent_id in agent_ids
    }

    test_dataset = build_image_dataset_from_split(
        extracted_dir=config.extracted_dir,
        index_path=config.index_path,
        split_manifest_path=config.train_test_manifest_path,
        split=config.test_split,
        image_size=config.image_size,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    initial_model = SimpleCNN(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
    )
    initial_state = initial_model.state_dict()

    agent_models: dict[str, Any] = {}
    for agent_id in agent_ids:
        model = SimpleCNN(
            num_classes=config.num_classes,
            input_channels=config.input_channels,
        )
        model.load_state_dict(initial_state)
        model.to(device)
        agent_models[agent_id] = model

    metrics_logger = MetricsCSVLogger(
        config.metrics_path,
        fieldnames=[
            "round",
            "mean_train_loss",
            "mean_train_accuracy",
            "mean_test_loss",
            "mean_test_accuracy",
            "total_train_examples",
            "num_agents",
            "num_encounters",
            "unpaired_agents",
        ],
    )

    metrics_rows: list[dict[str, float | int]] = []
    for round_index in range(1, config.rounds + 1):
        train_loss_sum = 0.0
        train_correct_sum = 0.0
        total_train_examples = 0

        for agent_id in agent_ids:
            agent_dataset = agent_datasets[agent_id]
            if len(agent_dataset) == 0:
                continue

            train_loader = DataLoader(
                agent_dataset,
                batch_size=config.batch_size,
                shuffle=True,
            )
            optimizer = torch.optim.SGD(
                agent_models[agent_id].parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
            )
            local_metrics = train_local_epochs(
                agent_models[agent_id],
                train_loader,
                optimizer,
                criterion,
                local_epochs=config.local_epochs,
                device=device,
            )
            final_metrics = local_metrics[-1]
            train_loss_sum += final_metrics.loss * final_metrics.num_examples
            train_correct_sum += final_metrics.accuracy * final_metrics.num_examples
            total_train_examples += final_metrics.num_examples

        if total_train_examples <= 0:
            raise RuntimeError(
                "No local updates were produced for any DFL agent. "
                "Check partition manifest and client datasets."
            )

        pairings, unpaired = sample_random_pairings(
            agent_ids,
            rng,
            max_pairs=config.encounters_per_round,
        )
        for left_agent_id, right_agent_id in pairings:
            averaged_state = weighted_average_state_dicts(
                [
                    agent_models[left_agent_id].state_dict(),
                    agent_models[right_agent_id].state_dict(),
                ],
                [1, 1],
            )
            agent_models[left_agent_id].load_state_dict(averaged_state)
            agent_models[right_agent_id].load_state_dict(averaged_state)

        test_losses: list[float] = []
        test_accuracies: list[float] = []
        for agent_id in agent_ids:
            agent_test_metrics = evaluate_model(
                agent_models[agent_id],
                test_loader,
                criterion,
                device=device,
            )
            test_losses.append(agent_test_metrics.loss)
            test_accuracies.append(agent_test_metrics.accuracy)

        mean_train_loss = train_loss_sum / float(total_train_examples)
        mean_train_accuracy = train_correct_sum / float(total_train_examples)
        mean_test_loss = float(sum(test_losses) / len(test_losses))
        mean_test_accuracy = float(sum(test_accuracies) / len(test_accuracies))

        metrics_row = _build_round_metrics_row(
            round_index=round_index,
            mean_train_loss=mean_train_loss,
            mean_train_accuracy=mean_train_accuracy,
            mean_test_loss=mean_test_loss,
            mean_test_accuracy=mean_test_accuracy,
            total_train_examples=total_train_examples,
            num_agents=len(agent_ids),
            num_encounters=len(pairings),
            unpaired_agents=len(unpaired),
        )
        metrics_logger.log(metrics_row)
        metrics_rows.append(metrics_row)

        for agent_index, agent_id in enumerate(agent_ids):
            snapshot_path = (
                config.models_dir / f"round_{round_index:03d}_agent_{agent_index:02d}.pkl"
            )
            save_model_snapshot(
                agent_models[agent_id],
                snapshot_path,
                model_name="simple_cnn",
                round_index=round_index,
                config=config.to_serializable_dict(),
                metrics={"agent_id": agent_id, **metrics_row},
            )

        print(
            f"[DFL] round={round_index:03d} "
            f"mean_test_accuracy={mean_test_accuracy:.4f} "
            f"mean_test_loss={mean_test_loss:.4f} "
            f"encounters={len(pairings)}"
        )

    save_accuracy_line_plot(
        rounds=[int(row["round"]) for row in metrics_rows],
        accuracies=[float(row["mean_test_accuracy"]) for row in metrics_rows],
        output_path=config.plot_path,
        title="DFL (No Cache) Mean Test Accuracy vs Communication Rounds",
    )

    print(f"Communication round definition: {COMMUNICATION_ROUND_DEFINITION}")
    print(f"DFL metrics written to: {config.metrics_path}")
    print(f"DFL snapshots written to: {config.models_dir}")
    print(f"DFL plot written to: {config.plot_path}")
    return metrics_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DFL baseline (without caching).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dfl.yaml"),
        help="Path to DFL config file (JSON-compatible YAML).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_dfl_config(args.config)
    run_dfl_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
