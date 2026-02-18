from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.fl.fedavg import FedAvgConfig, run_fedavg_experiment
from src.plotting.line_plot import save_accuracy_line_plot, save_multi_accuracy_line_plot

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any


@dataclass(frozen=True)
class FedAWAConfig:
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
    alignment_epsilon: float
    negative_alignment_mode: str
    run_fedavg_baseline: bool
    fedawa_metrics_path: Path
    fedawa_models_dir: Path
    fedawa_plot_path: Path
    fedavg_metrics_path: Path
    fedavg_models_dir: Path
    fedavg_plot_path: Path
    comparison_plot_path: Path
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
            "alignment_epsilon": self.alignment_epsilon,
            "negative_alignment_mode": self.negative_alignment_mode,
            "run_fedavg_baseline": self.run_fedavg_baseline,
            "fedawa_metrics_path": str(self.fedawa_metrics_path),
            "fedawa_models_dir": str(self.fedawa_models_dir),
            "fedawa_plot_path": str(self.fedawa_plot_path),
            "fedavg_metrics_path": str(self.fedavg_metrics_path),
            "fedavg_models_dir": str(self.fedavg_models_dir),
            "fedavg_plot_path": str(self.fedavg_plot_path),
            "comparison_plot_path": str(self.comparison_plot_path),
            "clean_artifacts": self.clean_artifacts,
        }


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for '{key}' in FedAWA config.")
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


def _validate_negative_alignment_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized != "clip_to_zero":
        raise ValueError(
            "fedawa.negative_alignment_mode must be 'clip_to_zero'. " f"Received: {mode!r}."
        )
    return normalized


def load_fedawa_config(config_path: Path) -> FedAWAConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        try:
            payload = json.load(config_file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "FedAWA config must be valid JSON (JSON is valid YAML). "
                f"Could not parse {config_path}."
            ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("FedAWA config root must be a mapping object.")

    dataset_cfg = _require_mapping(payload, "dataset")
    training_cfg = _require_mapping(payload, "training")
    model_cfg = _require_mapping(payload, "model")
    fedawa_cfg = _require_mapping(payload, "fedawa")
    artifacts_cfg = _require_mapping(payload, "artifacts")

    return FedAWAConfig(
        experiment_name=str(payload.get("experiment_name", "fedawa")),
        seed=int(payload.get("seed", 42)),
        device=str(payload.get("device", "cpu")),
        extracted_dir=Path(str(dataset_cfg["extracted_dir"])),
        index_path=Path(str(dataset_cfg["index_path"])),
        train_test_manifest_path=Path(str(dataset_cfg["train_test_manifest_path"])),
        partition_manifest_path=Path(str(dataset_cfg["partition_manifest_path"])),
        test_split=str(dataset_cfg.get("test_split", "test")),
        image_size=_parse_image_size(dataset_cfg.get("image_size", [64, 64])),
        num_clients=_validate_positive_int(
            "training.num_clients",
            int(training_cfg["num_clients"]),
        ),
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
        num_classes=_validate_positive_int("model.num_classes", int(model_cfg["num_classes"])),
        input_channels=_validate_positive_int(
            "model.input_channels",
            int(model_cfg.get("input_channels", 3)),
        ),
        alignment_epsilon=_validate_positive_float(
            "fedawa.alignment_epsilon",
            float(fedawa_cfg.get("alignment_epsilon", 1e-12)),
        ),
        negative_alignment_mode=_validate_negative_alignment_mode(
            str(fedawa_cfg.get("negative_alignment_mode", "clip_to_zero"))
        ),
        run_fedavg_baseline=bool(fedawa_cfg.get("run_fedavg_baseline", True)),
        fedawa_metrics_path=Path(str(artifacts_cfg["fedawa_metrics_path"])),
        fedawa_models_dir=Path(str(artifacts_cfg["fedawa_models_dir"])),
        fedawa_plot_path=Path(str(artifacts_cfg["fedawa_plot_path"])),
        fedavg_metrics_path=Path(str(artifacts_cfg["fedavg_metrics_path"])),
        fedavg_models_dir=Path(str(artifacts_cfg["fedavg_models_dir"])),
        fedavg_plot_path=Path(str(artifacts_cfg["fedavg_plot_path"])),
        comparison_plot_path=Path(str(artifacts_cfg["comparison_plot_path"])),
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


def _prepare_fedawa_artifact_paths(config: FedAWAConfig) -> None:
    config.fedawa_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.fedawa_models_dir.mkdir(parents=True, exist_ok=True)
    config.fedawa_plot_path.parent.mkdir(parents=True, exist_ok=True)
    config.comparison_plot_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.clean_artifacts:
        return

    if config.fedawa_metrics_path.exists():
        config.fedawa_metrics_path.unlink()
    if config.fedawa_plot_path.exists():
        config.fedawa_plot_path.unlink()
    if config.comparison_plot_path.exists():
        config.comparison_plot_path.unlink()
    for snapshot_path in config.fedawa_models_dir.glob("round_*.pkl"):
        snapshot_path.unlink()


def _extract_client_ids(config: FedAWAConfig) -> list[str]:
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


def _clone_state_dict_cpu(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _state_dict_difference(
    left_state_dict: Mapping[str, Tensor],
    right_state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    import torch

    if set(left_state_dict) != set(right_state_dict):
        raise ValueError("Both state dicts must contain exactly the same keys.")

    return {
        name: left_state_dict[name].detach().cpu().to(torch.float32)
        - right_state_dict[name].detach().cpu().to(torch.float32)
        for name in left_state_dict
    }


def _state_dict_l2_norm(state_dict: Mapping[str, Tensor], *, epsilon: float) -> float:
    import torch

    norm_squared = 0.0
    for tensor in state_dict.values():
        tensor_fp32 = tensor.detach().cpu().to(torch.float32)
        norm_squared += float(torch.sum(tensor_fp32 * tensor_fp32).item())
    if norm_squared <= epsilon:
        return 0.0
    return math.sqrt(norm_squared)


def _cosine_similarity_between_state_deltas(
    left_delta: Mapping[str, Tensor],
    right_delta: Mapping[str, Tensor],
    *,
    epsilon: float,
) -> float:
    import torch

    if set(left_delta) != set(right_delta):
        raise ValueError("Delta state dicts must contain exactly the same keys.")

    dot_product = 0.0
    left_norm_squared = 0.0
    right_norm_squared = 0.0
    for name in left_delta:
        left_tensor = left_delta[name].detach().cpu().to(torch.float32)
        right_tensor = right_delta[name].detach().cpu().to(torch.float32)
        dot_product += float(torch.sum(left_tensor * right_tensor).item())
        left_norm_squared += float(torch.sum(left_tensor * left_tensor).item())
        right_norm_squared += float(torch.sum(right_tensor * right_tensor).item())

    if left_norm_squared <= epsilon or right_norm_squared <= epsilon:
        return 0.0

    denominator = math.sqrt(left_norm_squared) * math.sqrt(right_norm_squared) + epsilon
    return dot_product / denominator


def _dataset_size_weights(client_num_examples: Sequence[int]) -> list[float]:
    if not client_num_examples:
        raise ValueError("client_num_examples must not be empty.")
    if any(num_examples <= 0 for num_examples in client_num_examples):
        raise ValueError("All client_num_examples entries must be > 0.")

    total_examples = float(sum(client_num_examples))
    return [float(num_examples) / total_examples for num_examples in client_num_examples]


def compute_fedawa_client_weights(
    *,
    cosine_alignments: Sequence[float] | None,
    client_num_examples: Sequence[int],
    negative_alignment_mode: str = "clip_to_zero",
    epsilon: float = 1e-12,
) -> list[float]:
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    base_weights = _dataset_size_weights(client_num_examples)
    if cosine_alignments is None:
        return base_weights
    if len(cosine_alignments) != len(client_num_examples):
        raise ValueError("cosine_alignments and client_num_examples must have same length.")

    if negative_alignment_mode != "clip_to_zero":
        raise ValueError(
            "Only 'clip_to_zero' is supported for negative_alignment_mode, "
            f"got {negative_alignment_mode!r}."
        )

    raw_weights: list[float] = []
    for alignment, base_weight in zip(cosine_alignments, base_weights, strict=True):
        alignment_value = float(alignment)
        if not math.isfinite(alignment_value):
            raise ValueError(f"Alignment must be finite, got {alignment_value}.")
        raw_weights.append(base_weight * max(0.0, alignment_value))

    total_raw = float(sum(raw_weights))
    if total_raw <= epsilon:
        return base_weights
    return [raw / total_raw for raw in raw_weights]


def average_state_dicts_with_weights(
    client_state_dicts: Sequence[Mapping[str, Tensor]],
    client_weights: Sequence[float],
) -> dict[str, Tensor]:
    import torch

    if not client_state_dicts:
        raise ValueError("client_state_dicts must not be empty.")
    if len(client_state_dicts) != len(client_weights):
        raise ValueError("client_state_dicts and client_weights must have same length.")
    if any(weight < 0.0 for weight in client_weights):
        raise ValueError("All client_weights entries must be >= 0.")

    total_weight = float(sum(client_weights))
    if total_weight <= 0.0:
        raise ValueError("Sum of client_weights must be > 0.")
    normalized_weights = [float(weight) / total_weight for weight in client_weights]

    expected_keys = list(client_state_dicts[0].keys())
    expected_key_set = set(expected_keys)
    for client_index, state_dict in enumerate(client_state_dicts):
        if set(state_dict.keys()) != expected_key_set:
            raise ValueError(
                "All client state_dicts must contain exactly the same keys. "
                f"Mismatch at client index {client_index}."
            )

    aggregated: dict[str, Tensor] = {}
    for name in expected_keys:
        reference_tensor = client_state_dicts[0][name].detach().cpu()
        accumulator = torch.zeros_like(reference_tensor, dtype=torch.float32)
        for state_dict, weight in zip(client_state_dicts, normalized_weights, strict=True):
            accumulator += state_dict[name].detach().cpu().to(torch.float32) * weight

        if reference_tensor.is_floating_point():
            aggregated[name] = accumulator.to(dtype=reference_tensor.dtype)
        else:
            aggregated[name] = accumulator.round().to(dtype=reference_tensor.dtype)

    return aggregated


def _build_round_metrics_row(
    *,
    round_index: int,
    train_loss: float,
    train_accuracy: float,
    test_loss: float,
    test_accuracy: float,
    participating_clients: int,
    total_client_examples: int,
    alignment_mean: float,
    alignment_min: float,
    alignment_max: float,
    non_positive_alignments: int,
    direction_norm: float,
) -> dict[str, float | int]:
    return {
        "round": round_index,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "participating_clients": participating_clients,
        "total_client_examples": total_client_examples,
        "alignment_mean": alignment_mean,
        "alignment_min": alignment_min,
        "alignment_max": alignment_max,
        "non_positive_alignments": non_positive_alignments,
        "direction_norm": direction_norm,
    }


def _build_fedavg_config_from_fedawa(config: FedAWAConfig) -> FedAvgConfig:
    return FedAvgConfig(
        experiment_name="fedavg",
        seed=config.seed,
        device=config.device,
        extracted_dir=config.extracted_dir,
        index_path=config.index_path,
        train_test_manifest_path=config.train_test_manifest_path,
        partition_manifest_path=config.partition_manifest_path,
        test_split=config.test_split,
        image_size=config.image_size,
        num_clients=config.num_clients,
        dirichlet_alpha=config.dirichlet_alpha,
        rounds=config.rounds,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        metrics_path=config.fedavg_metrics_path,
        models_dir=config.fedavg_models_dir,
        plot_path=config.fedavg_plot_path,
        clean_artifacts=config.clean_artifacts,
    )


def _load_round_accuracy_series(metrics_path: Path) -> tuple[list[int], list[float]]:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"FedAvg metrics not found at {metrics_path}. "
            "Run FedAvg baseline first or enable fedawa.run_fedavg_baseline."
        )

    rounds: list[int] = []
    accuracies: list[float] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        reader = csv.DictReader(metrics_file)
        for row in reader:
            if "round" not in row or "test_accuracy" not in row:
                raise ValueError(
                    f"Metrics file {metrics_path} must contain 'round' and 'test_accuracy' columns."
                )
            rounds.append(int(float(str(row["round"]))))
            accuracies.append(float(str(row["test_accuracy"])))

    if not rounds:
        raise ValueError(f"Metrics file at {metrics_path} contains no rows.")
    return rounds, accuracies


def _extract_round_accuracy_from_rows(
    rows: Sequence[Mapping[str, float | int]],
) -> tuple[list[int], list[float]]:
    rounds: list[int] = []
    accuracies: list[float] = []
    for row in rows:
        rounds.append(int(row["round"]))
        accuracies.append(float(row["test_accuracy"]))
    return rounds, accuracies


def run_fedawa_experiment(config: FedAWAConfig) -> list[dict[str, float | int]]:
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

    _prepare_fedawa_artifact_paths(config)

    if config.run_fedavg_baseline:
        fedavg_rows = run_fedavg_experiment(_build_fedavg_config_from_fedawa(config))
        fedavg_rounds, fedavg_accuracies = _extract_round_accuracy_from_rows(fedavg_rows)
    else:
        fedavg_rounds, fedavg_accuracies = _load_round_accuracy_series(config.fedavg_metrics_path)

    set_global_seed(config.seed)
    device = _resolve_device(config.device)

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
        config.fedawa_metrics_path,
        fieldnames=[
            "round",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "participating_clients",
            "total_client_examples",
            "alignment_mean",
            "alignment_min",
            "alignment_max",
            "non_positive_alignments",
            "direction_norm",
        ],
    )

    metrics_rows: list[dict[str, float | int]] = []
    previous_global_state: dict[str, Tensor] | None = None

    for round_index in range(1, config.rounds + 1):
        global_state_before = _clone_state_dict_cpu(global_model.state_dict())
        direction_state = (
            None
            if previous_global_state is None
            else _state_dict_difference(global_state_before, previous_global_state)
        )
        direction_norm = (
            0.0
            if direction_state is None
            else _state_dict_l2_norm(direction_state, epsilon=config.alignment_epsilon)
        )

        client_state_dicts: list[dict[str, Tensor]] = []
        client_num_examples: list[int] = []
        cosine_alignments: list[float] = []
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
            local_state = _clone_state_dict_cpu(local_model.state_dict())

            client_state_dicts.append(local_state)
            client_num_examples.append(final_metrics.num_examples)
            train_loss_sum += final_metrics.loss * final_metrics.num_examples
            train_correct_sum += final_metrics.accuracy * final_metrics.num_examples

            if direction_state is None:
                cosine_alignments.append(0.0)
            else:
                tau_state = _state_dict_difference(local_state, global_state_before)
                cosine_alignments.append(
                    _cosine_similarity_between_state_deltas(
                        tau_state,
                        direction_state,
                        epsilon=config.alignment_epsilon,
                    )
                )

        if not client_state_dicts:
            raise RuntimeError(
                "No client updates were produced. Check partition manifest and client datasets."
            )

        client_weights = compute_fedawa_client_weights(
            cosine_alignments=cosine_alignments,
            client_num_examples=client_num_examples,
            negative_alignment_mode=config.negative_alignment_mode,
            epsilon=config.alignment_epsilon,
        )
        aggregated_state = average_state_dicts_with_weights(client_state_dicts, client_weights)
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
            alignment_mean=float(sum(cosine_alignments) / len(cosine_alignments)),
            alignment_min=float(min(cosine_alignments)),
            alignment_max=float(max(cosine_alignments)),
            non_positive_alignments=int(sum(1 for value in cosine_alignments if value <= 0.0)),
            direction_norm=direction_norm,
        )
        metrics_logger.log(metrics_row)
        metrics_rows.append(metrics_row)

        snapshot_path = config.fedawa_models_dir / f"round_{round_index}.pkl"
        save_model_snapshot(
            global_model,
            snapshot_path,
            model_name="simple_cnn",
            round_index=round_index,
            config=config.to_serializable_dict(),
            metrics=metrics_row,
        )

        print(
            f"[FedAWA] round={round_index:03d} "
            f"test_accuracy={test_metrics.accuracy:.4f} "
            f"test_loss={test_metrics.loss:.4f} "
            f"alignment_mean={metrics_row['alignment_mean']:.4f}"
        )
        previous_global_state = global_state_before

    fedawa_rounds, fedawa_accuracies = _extract_round_accuracy_from_rows(metrics_rows)
    save_accuracy_line_plot(
        rounds=fedawa_rounds,
        accuracies=fedawa_accuracies,
        output_path=config.fedawa_plot_path,
        title="FedAWA Test Accuracy vs Communication Rounds",
    )
    save_multi_accuracy_line_plot(
        series=[
            ("FedAvg", fedavg_rounds, fedavg_accuracies),
            ("FedAWA", fedawa_rounds, fedawa_accuracies),
        ],
        output_path=config.comparison_plot_path,
        title="FedAvg vs FedAWA Test Accuracy",
    )

    print(f"FedAWA metrics written to: {config.fedawa_metrics_path}")
    print(f"FedAWA snapshots written to: {config.fedawa_models_dir}")
    print(f"FedAWA plot written to: {config.fedawa_plot_path}")
    print(f"FedAvg vs FedAWA plot written to: {config.comparison_plot_path}")
    return metrics_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FedAWA experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fedawa.yaml"),
        help="Path to FedAWA config file (JSON-compatible YAML).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_fedawa_config(args.config)
    run_fedawa_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
