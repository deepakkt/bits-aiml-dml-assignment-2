from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.plotting.line_plot import save_accuracy_line_plot

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class FedAvgConfig:
    experiment_name: str
    seed: int
    device: str
    extracted_dir: Path
    index_path: Path
    train_test_manifest_path: Path
    partition_manifest_path: Path
    test_split: str
    image_size: tuple[int, int]
    num_clients: int
    dirichlet_alpha: float
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
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
            "num_clients": self.num_clients,
            "dirichlet_alpha": self.dirichlet_alpha,
            "rounds": self.rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "metrics_path": str(self.metrics_path),
            "models_dir": str(self.models_dir),
            "plot_path": str(self.plot_path),
            "clean_artifacts": self.clean_artifacts,
        }


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for '{key}' in FedAvg config.")
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


def load_fedavg_config(config_path: Path) -> FedAvgConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        try:
            payload = json.load(config_file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "FedAvg config must be valid JSON (JSON is valid YAML). "
                f"Could not parse {config_path}."
            ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("FedAvg config root must be a mapping object.")

    dataset_cfg = _require_mapping(payload, "dataset")
    fedavg_cfg = _require_mapping(payload, "fedavg")
    model_cfg = _require_mapping(payload, "model")
    artifacts_cfg = _require_mapping(payload, "artifacts")

    return FedAvgConfig(
        experiment_name=str(payload.get("experiment_name", "fedavg")),
        seed=int(payload.get("seed", 42)),
        device=str(payload.get("device", "cpu")),
        extracted_dir=Path(str(dataset_cfg["extracted_dir"])),
        index_path=Path(str(dataset_cfg["index_path"])),
        train_test_manifest_path=Path(str(dataset_cfg["train_test_manifest_path"])),
        partition_manifest_path=Path(str(dataset_cfg["partition_manifest_path"])),
        test_split=str(dataset_cfg.get("test_split", "test")),
        image_size=_parse_image_size(dataset_cfg.get("image_size", [64, 64])),
        num_clients=_validate_positive_int("fedavg.num_clients", int(fedavg_cfg["num_clients"])),
        dirichlet_alpha=_validate_positive_float(
            "fedavg.dirichlet_alpha",
            float(fedavg_cfg.get("dirichlet_alpha", 0.5)),
        ),
        rounds=_validate_positive_int("fedavg.rounds", int(fedavg_cfg["rounds"])),
        local_epochs=_validate_positive_int("fedavg.local_epochs", int(fedavg_cfg["local_epochs"])),
        batch_size=_validate_positive_int("fedavg.batch_size", int(fedavg_cfg["batch_size"])),
        learning_rate=_validate_positive_float(
            "fedavg.learning_rate",
            float(fedavg_cfg["learning_rate"]),
        ),
        momentum=float(fedavg_cfg.get("momentum", 0.0)),
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


def weighted_average_state_dicts(
    client_state_dicts: Sequence[Mapping[str, Tensor]],
    client_num_examples: Sequence[int],
) -> dict[str, Tensor]:
    import torch

    if not client_state_dicts:
        raise ValueError("client_state_dicts must not be empty.")
    if len(client_state_dicts) != len(client_num_examples):
        raise ValueError("client_state_dicts and client_num_examples must have same length.")
    if any(num_examples <= 0 for num_examples in client_num_examples):
        raise ValueError("All client_num_examples entries must be > 0.")

    expected_keys = list(client_state_dicts[0].keys())
    expected_key_set = set(expected_keys)
    for client_index, state_dict in enumerate(client_state_dicts):
        current_keys = set(state_dict.keys())
        if current_keys != expected_key_set:
            raise ValueError(
                "All client state_dicts must contain exactly the same keys. "
                f"Mismatch at client index {client_index}."
            )

    total_examples = float(sum(client_num_examples))
    aggregated: dict[str, Tensor] = {}

    for name in expected_keys:
        reference_tensor = client_state_dicts[0][name].detach().cpu()
        accumulator = torch.zeros_like(reference_tensor, dtype=torch.float32)
        for state_dict, num_examples in zip(client_state_dicts, client_num_examples, strict=True):
            weight = float(num_examples) / total_examples
            accumulator += state_dict[name].detach().cpu().to(torch.float32) * weight

        if reference_tensor.is_floating_point():
            aggregated[name] = accumulator.to(dtype=reference_tensor.dtype)
        else:
            aggregated[name] = accumulator.round().to(dtype=reference_tensor.dtype)

    return aggregated


def _resolve_device(requested_device: str) -> Any:
    import torch

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested_device}' but CUDA is unavailable in this environment."
        )
    return torch.device(requested_device)


def _prepare_artifact_paths(config: FedAvgConfig) -> None:
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


def _extract_client_ids(config: FedAvgConfig) -> list[str]:
    from src.data.dataset import read_split_manifest

    partition_manifest = read_split_manifest(config.partition_manifest_path)

    partition_num_clients = int(partition_manifest.get("num_clients", -1))
    if partition_num_clients != config.num_clients:
        raise ValueError(
            "Client count mismatch between config and partition manifest: "
            f"{config.num_clients=} vs {partition_num_clients=}."
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

    client_ids = sorted(str(client_id) for client_id in clients)
    if len(client_ids) != config.num_clients:
        raise ValueError(
            f"Expected {config.num_clients} client entries but found {len(client_ids)}."
        )
    return client_ids


def _build_round_metrics_row(
    *,
    round_index: int,
    train_loss: float,
    train_accuracy: float,
    test_loss: float,
    test_accuracy: float,
    participating_clients: int,
    total_client_examples: int,
) -> dict[str, float | int]:
    return {
        "round": round_index,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "participating_clients": participating_clients,
        "total_client_examples": total_client_examples,
    }


def run_fedavg_experiment(config: FedAvgConfig) -> list[dict[str, float | int]]:
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
    device = _resolve_device(config.device)
    _prepare_artifact_paths(config)

    client_ids = _extract_client_ids(config)
    client_datasets = {
        client_id: build_image_dataset_from_client(
            extracted_dir=config.extracted_dir,
            index_path=config.index_path,
            partition_manifest_path=config.partition_manifest_path,
            client_id=client_id,
            image_size=config.image_size,
        )
        for client_id in client_ids
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
    global_model = SimpleCNN(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
    )
    global_model.to(device)

    metrics_logger = MetricsCSVLogger(
        config.metrics_path,
        fieldnames=[
            "round",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "participating_clients",
            "total_client_examples",
        ],
    )

    metrics_rows: list[dict[str, float | int]] = []
    for round_index in range(1, config.rounds + 1):
        client_state_dicts: list[dict[str, Tensor]] = []
        client_num_examples: list[int] = []
        train_loss_sum = 0.0
        train_correct_sum = 0.0

        for client_id in client_ids:
            client_dataset = client_datasets[client_id]
            if len(client_dataset) == 0:
                continue

            data_loader = DataLoader(
                client_dataset,
                batch_size=config.batch_size,
                shuffle=True,
            )

            local_model = SimpleCNN(
                num_classes=config.num_classes,
                input_channels=config.input_channels,
            )
            local_model.load_state_dict(global_model.state_dict())

            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
            )
            local_metrics = train_local_epochs(
                local_model,
                data_loader,
                optimizer,
                criterion,
                local_epochs=config.local_epochs,
                device=device,
            )
            final_metrics = local_metrics[-1]

            client_state_dicts.append(dict(local_model.state_dict()))
            client_num_examples.append(final_metrics.num_examples)
            train_loss_sum += final_metrics.loss * final_metrics.num_examples
            train_correct_sum += final_metrics.accuracy * final_metrics.num_examples

        if not client_state_dicts:
            raise RuntimeError(
                "No client updates were produced. Check partition manifest and client datasets."
            )

        aggregated_state = weighted_average_state_dicts(client_state_dicts, client_num_examples)
        global_model.load_state_dict(aggregated_state)

        test_metrics = evaluate_model(global_model, test_loader, criterion, device=device)
        total_client_examples = int(sum(client_num_examples))
        train_loss = train_loss_sum / float(total_client_examples)
        train_accuracy = train_correct_sum / float(total_client_examples)
        metrics_row = _build_round_metrics_row(
            round_index=round_index,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            test_loss=test_metrics.loss,
            test_accuracy=test_metrics.accuracy,
            participating_clients=len(client_num_examples),
            total_client_examples=total_client_examples,
        )
        metrics_logger.log(metrics_row)
        metrics_rows.append(metrics_row)

        snapshot_path = config.models_dir / f"round_{round_index}.pkl"
        save_model_snapshot(
            global_model,
            snapshot_path,
            model_name="simple_cnn",
            round_index=round_index,
            config=config.to_serializable_dict(),
            metrics=metrics_row,
        )

        print(
            f"[FedAvg] round={round_index:03d} "
            f"test_accuracy={test_metrics.accuracy:.4f} "
            f"test_loss={test_metrics.loss:.4f}"
        )

    save_accuracy_line_plot(
        rounds=[int(row["round"]) for row in metrics_rows],
        accuracies=[float(row["test_accuracy"]) for row in metrics_rows],
        output_path=config.plot_path,
        title="FedAvg Test Accuracy vs Communication Rounds",
    )
    print(f"FedAvg metrics written to: {config.metrics_path}")
    print(f"FedAvg snapshots written to: {config.models_dir}")
    print(f"FedAvg plot written to: {config.plot_path}")
    return metrics_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FedAvg experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fedavg.yaml"),
        help="Path to FedAvg config file (JSON-compatible YAML).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_fedavg_config(args.config)
    run_fedavg_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
