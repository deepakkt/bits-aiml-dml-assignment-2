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

from src.dfl.simulator import DFLConfig, run_dfl_experiment, sample_random_pairings
from src.fl.fedavg import weighted_average_state_dicts
from src.plotting.line_plot import save_accuracy_line_plot, save_multi_accuracy_line_plot

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any

CACHED_COMMUNICATION_ROUND_DEFINITION = (
    "One communication round includes (1) stale-cache eviction and local model warm-start "
    "by averaging the agent model with all cached models, (2) local training on each "
    "agent's private data, followed by (3) random pairwise encounters where encountered "
    "agents average their current models and cache each peer's pre-encounter model."
)


@dataclass(frozen=True)
class CachedDFLConfig:
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
    cache_size: int
    tau_max: int
    run_no_cache_baseline: bool
    num_classes: int
    input_channels: int
    cached_metrics_path: Path
    cached_models_dir: Path
    cached_plot_path: Path
    no_cache_metrics_path: Path
    no_cache_models_dir: Path
    no_cache_plot_path: Path
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
            "num_agents": self.num_agents,
            "dirichlet_alpha": self.dirichlet_alpha,
            "rounds": self.rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "encounters_per_round": self.encounters_per_round,
            "cache_size": self.cache_size,
            "tau_max": self.tau_max,
            "run_no_cache_baseline": self.run_no_cache_baseline,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "cached_metrics_path": str(self.cached_metrics_path),
            "cached_models_dir": str(self.cached_models_dir),
            "cached_plot_path": str(self.cached_plot_path),
            "no_cache_metrics_path": str(self.no_cache_metrics_path),
            "no_cache_models_dir": str(self.no_cache_models_dir),
            "no_cache_plot_path": str(self.no_cache_plot_path),
            "comparison_plot_path": str(self.comparison_plot_path),
            "clean_artifacts": self.clean_artifacts,
            "communication_round_definition": CACHED_COMMUNICATION_ROUND_DEFINITION,
        }


@dataclass(frozen=True)
class CachedModelEntry:
    source_agent_id: str
    round_seen: int
    state_dict: dict[str, Tensor]


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping for '{key}' in Cached-DFL config.")
    return value


def _validate_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _validate_non_negative_int(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
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


def load_cached_dfl_config(config_path: Path) -> CachedDFLConfig:
    with config_path.open("r", encoding="utf-8") as config_file:
        try:
            payload = json.load(config_file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Cached-DFL config must be valid JSON (JSON is valid YAML). "
                f"Could not parse {config_path}."
            ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("Cached-DFL config root must be a mapping object.")

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

    return CachedDFLConfig(
        experiment_name=str(payload.get("experiment_name", "cached_dfl")),
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
        cache_size=_validate_positive_int("dfl.cache_size", int(dfl_cfg.get("cache_size", 3))),
        tau_max=_validate_non_negative_int("dfl.tau_max", int(dfl_cfg.get("tau_max", 5))),
        run_no_cache_baseline=bool(dfl_cfg.get("run_no_cache_baseline", True)),
        num_classes=_validate_positive_int("model.num_classes", int(model_cfg["num_classes"])),
        input_channels=_validate_positive_int(
            "model.input_channels",
            int(model_cfg.get("input_channels", 3)),
        ),
        cached_metrics_path=Path(str(artifacts_cfg["cached_metrics_path"])),
        cached_models_dir=Path(str(artifacts_cfg["cached_models_dir"])),
        cached_plot_path=Path(str(artifacts_cfg["cached_plot_path"])),
        no_cache_metrics_path=Path(str(artifacts_cfg["no_cache_metrics_path"])),
        no_cache_models_dir=Path(str(artifacts_cfg["no_cache_models_dir"])),
        no_cache_plot_path=Path(str(artifacts_cfg["no_cache_plot_path"])),
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


def _prepare_cached_artifact_paths(config: CachedDFLConfig) -> None:
    config.cached_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.cached_models_dir.mkdir(parents=True, exist_ok=True)
    config.cached_plot_path.parent.mkdir(parents=True, exist_ok=True)
    config.comparison_plot_path.parent.mkdir(parents=True, exist_ok=True)

    if not config.clean_artifacts:
        return

    if config.cached_metrics_path.exists():
        config.cached_metrics_path.unlink()
    if config.cached_plot_path.exists():
        config.cached_plot_path.unlink()
    if config.comparison_plot_path.exists():
        config.comparison_plot_path.unlink()
    for snapshot_path in config.cached_models_dir.glob("round_*.pkl"):
        snapshot_path.unlink()


def _extract_agent_ids(config: CachedDFLConfig) -> list[str]:
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


def _clone_state_dict_cpu(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def evict_stale_cache_entries(
    cache_entries: Sequence[CachedModelEntry],
    *,
    current_round: int,
    tau_max: int,
) -> tuple[list[CachedModelEntry], int]:
    if tau_max < 0:
        raise ValueError(f"tau_max must be >= 0, got {tau_max}")

    retained: list[CachedModelEntry] = []
    evicted = 0
    for entry in cache_entries:
        staleness = current_round - entry.round_seen
        if staleness <= tau_max:
            retained.append(entry)
        else:
            evicted += 1
    return retained, evicted


def upsert_cache_entry(
    cache_entries: Sequence[CachedModelEntry],
    *,
    new_entry: CachedModelEntry,
    cache_size: int,
) -> tuple[list[CachedModelEntry], int]:
    if cache_size <= 0:
        raise ValueError(f"cache_size must be > 0, got {cache_size}")

    deduplicated = [
        entry for entry in cache_entries if entry.source_agent_id != new_entry.source_agent_id
    ]
    deduplicated.append(new_entry)
    deduplicated.sort(key=lambda entry: (entry.round_seen, entry.source_agent_id), reverse=True)

    retained = deduplicated[:cache_size]
    capacity_evictions = len(deduplicated) - len(retained)
    return retained, capacity_evictions


def aggregate_local_and_cached_state_dicts(
    *,
    local_state_dict: Mapping[str, Tensor],
    cache_entries: Sequence[CachedModelEntry],
) -> dict[str, Tensor]:
    if not cache_entries:
        return _clone_state_dict_cpu(local_state_dict)

    return weighted_average_state_dicts(
        [local_state_dict, *[entry.state_dict for entry in cache_entries]],
        [1] * (len(cache_entries) + 1),
    )


def _build_no_cache_dfl_config(config: CachedDFLConfig) -> DFLConfig:
    return DFLConfig(
        experiment_name="dfl",
        seed=config.seed,
        device=config.device,
        extracted_dir=config.extracted_dir,
        index_path=config.index_path,
        train_test_manifest_path=config.train_test_manifest_path,
        partition_manifest_path=config.partition_manifest_path,
        test_split=config.test_split,
        image_size=config.image_size,
        num_agents=config.num_agents,
        dirichlet_alpha=config.dirichlet_alpha,
        rounds=config.rounds,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        encounters_per_round=config.encounters_per_round,
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        metrics_path=config.no_cache_metrics_path,
        models_dir=config.no_cache_models_dir,
        plot_path=config.no_cache_plot_path,
        clean_artifacts=config.clean_artifacts,
    )


def _load_round_accuracy_series(metrics_path: Path) -> tuple[list[int], list[float]]:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No-cache DFL metrics not found at {metrics_path}. "
            "Run baseline DFL first or enable dfl.run_no_cache_baseline."
        )

    rounds: list[int] = []
    accuracies: list[float] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        reader = csv.DictReader(metrics_file)
        for row in reader:
            if "round" not in row or "mean_test_accuracy" not in row:
                raise ValueError(
                    f"Metrics file {metrics_path} must contain "
                    "'round' and 'mean_test_accuracy' columns."
                )
            rounds.append(int(float(str(row["round"]))))
            accuracies.append(float(str(row["mean_test_accuracy"])))

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
        accuracies.append(float(row["mean_test_accuracy"]))
    return rounds, accuracies


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
    agents_with_cache: int,
    mean_cache_entries_before_local_update: float,
    stale_evictions: int,
    capacity_evictions: int,
    cached_models_received: int,
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
        "agents_with_cache": agents_with_cache,
        "mean_cache_entries_before_local_update": mean_cache_entries_before_local_update,
        "stale_evictions": stale_evictions,
        "capacity_evictions": capacity_evictions,
        "cached_models_received": cached_models_received,
    }


def run_cached_dfl_experiment(config: CachedDFLConfig) -> list[dict[str, float | int]]:
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

    _prepare_cached_artifact_paths(config)

    if config.run_no_cache_baseline:
        dfl_rows = run_dfl_experiment(_build_no_cache_dfl_config(config))
        no_cache_rounds, no_cache_accuracies = _extract_round_accuracy_from_rows(dfl_rows)
    else:
        no_cache_rounds, no_cache_accuracies = _load_round_accuracy_series(
            config.no_cache_metrics_path
        )

    set_global_seed(config.seed)
    rng = random.Random(config.seed)
    device = _resolve_device(config.device)

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
    agent_caches: dict[str, list[CachedModelEntry]] = {agent_id: [] for agent_id in agent_ids}
    for agent_id in agent_ids:
        model = SimpleCNN(
            num_classes=config.num_classes,
            input_channels=config.input_channels,
        )
        model.load_state_dict(initial_state)
        model.to(device)
        agent_models[agent_id] = model

    metrics_logger = MetricsCSVLogger(
        config.cached_metrics_path,
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
            "agents_with_cache",
            "mean_cache_entries_before_local_update",
            "stale_evictions",
            "capacity_evictions",
            "cached_models_received",
        ],
    )

    metrics_rows: list[dict[str, float | int]] = []
    for round_index in range(1, config.rounds + 1):
        train_loss_sum = 0.0
        train_correct_sum = 0.0
        total_train_examples = 0

        total_cache_entries_before_local_update = 0
        stale_evictions = 0
        capacity_evictions = 0
        cached_models_received = 0
        agents_with_cache = 0

        for agent_id in agent_ids:
            fresh_cache_entries, stale_count = evict_stale_cache_entries(
                agent_caches[agent_id],
                current_round=round_index,
                tau_max=config.tau_max,
            )
            agent_caches[agent_id] = fresh_cache_entries
            stale_evictions += stale_count
            total_cache_entries_before_local_update += len(fresh_cache_entries)
            if fresh_cache_entries:
                agents_with_cache += 1

            merged_state = aggregate_local_and_cached_state_dicts(
                local_state_dict=_clone_state_dict_cpu(agent_models[agent_id].state_dict()),
                cache_entries=fresh_cache_entries,
            )
            agent_models[agent_id].load_state_dict(merged_state)

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
                "No local updates were produced for any Cached-DFL agent. "
                "Check partition manifest and client datasets."
            )

        pairings, unpaired = sample_random_pairings(
            agent_ids,
            rng,
            max_pairs=config.encounters_per_round,
        )
        for left_agent_id, right_agent_id in pairings:
            left_state_before = _clone_state_dict_cpu(agent_models[left_agent_id].state_dict())
            right_state_before = _clone_state_dict_cpu(agent_models[right_agent_id].state_dict())

            averaged_state = weighted_average_state_dicts(
                [left_state_before, right_state_before],
                [1, 1],
            )
            agent_models[left_agent_id].load_state_dict(averaged_state)
            agent_models[right_agent_id].load_state_dict(averaged_state)

            left_cache, left_capacity_evictions = upsert_cache_entry(
                agent_caches[left_agent_id],
                new_entry=CachedModelEntry(
                    source_agent_id=right_agent_id,
                    round_seen=round_index,
                    state_dict=right_state_before,
                ),
                cache_size=config.cache_size,
            )
            right_cache, right_capacity_evictions = upsert_cache_entry(
                agent_caches[right_agent_id],
                new_entry=CachedModelEntry(
                    source_agent_id=left_agent_id,
                    round_seen=round_index,
                    state_dict=left_state_before,
                ),
                cache_size=config.cache_size,
            )
            agent_caches[left_agent_id] = left_cache
            agent_caches[right_agent_id] = right_cache
            capacity_evictions += left_capacity_evictions + right_capacity_evictions
            cached_models_received += 2

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
            agents_with_cache=agents_with_cache,
            mean_cache_entries_before_local_update=(
                total_cache_entries_before_local_update / float(len(agent_ids))
            ),
            stale_evictions=stale_evictions,
            capacity_evictions=capacity_evictions,
            cached_models_received=cached_models_received,
        )
        metrics_logger.log(metrics_row)
        metrics_rows.append(metrics_row)

        for agent_index, agent_id in enumerate(agent_ids):
            snapshot_path = (
                config.cached_models_dir / f"round_{round_index:03d}_agent_{agent_index:02d}.pkl"
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
            f"[Cached-DFL] round={round_index:03d} "
            f"mean_test_accuracy={mean_test_accuracy:.4f} "
            f"mean_test_loss={mean_test_loss:.4f} "
            f"cache_mean={metrics_row['mean_cache_entries_before_local_update']:.2f}"
        )

    cached_rounds, cached_accuracies = _extract_round_accuracy_from_rows(metrics_rows)
    save_accuracy_line_plot(
        rounds=cached_rounds,
        accuracies=cached_accuracies,
        output_path=config.cached_plot_path,
        title=(
            f"Cached-DFL (cache_size={config.cache_size}, tau_max={config.tau_max}) "
            "Mean Test Accuracy vs Communication Rounds"
        ),
    )
    save_multi_accuracy_line_plot(
        series=[
            ("DFL (No Cache)", no_cache_rounds, no_cache_accuracies),
            ("Cached-DFL", cached_rounds, cached_accuracies),
        ],
        output_path=config.comparison_plot_path,
        title="DFL (No Cache) vs Cached-DFL Test Accuracy",
    )

    print(f"Cached-DFL communication round definition: {CACHED_COMMUNICATION_ROUND_DEFINITION}")
    print(f"Cached-DFL metrics written to: {config.cached_metrics_path}")
    print(f"Cached-DFL snapshots written to: {config.cached_models_dir}")
    print(f"Cached-DFL plot written to: {config.cached_plot_path}")
    print(f"DFL vs Cached-DFL plot written to: {config.comparison_plot_path}")
    return metrics_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cached-DFL experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cached_dfl.yaml"),
        help="Path to Cached-DFL config file (JSON-compatible YAML).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_cached_dfl_config(args.config)
    run_cached_dfl_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
