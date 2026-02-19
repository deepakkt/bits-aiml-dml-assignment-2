from __future__ import annotations

import json
import random
from pathlib import Path

import pytest


def test_load_dfl_config_from_json_compatible_yaml(tmp_path: Path) -> None:
    from src.dfl.simulator import load_dfl_config

    config_path = tmp_path / "dfl.yaml"
    payload = {
        "experiment_name": "dfl-test",
        "seed": 21,
        "dataset": {
            "extracted_dir": "data/extracted",
            "index_path": "data/processed/image_index.jsonl",
            "train_test_manifest_path": "data/splits/train_test_manifest.json",
            "partition_manifest_path": "data/splits/dirichlet_clients_manifest.json",
            "image_size": [64, 64],
        },
        "training": {
            "num_agents": 10,
            "dirichlet_alpha": 0.5,
            "rounds": 3,
            "local_epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.01,
        },
        "dfl": {
            "encounters_per_round": 5,
        },
        "model": {
            "num_classes": 2,
            "input_channels": 3,
        },
        "artifacts": {
            "metrics_path": "artifacts/metrics/dfl/metrics.csv",
            "models_dir": "artifacts/models/dfl",
            "plot_path": "artifacts/plots/dfl/accuracy_vs_rounds.png",
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_dfl_config(config_path)

    assert config.experiment_name == "dfl-test"
    assert config.seed == 21
    assert config.num_agents == 10
    assert config.encounters_per_round == 5


def test_load_dfl_config_rejects_too_many_pairs(tmp_path: Path) -> None:
    from src.dfl.simulator import load_dfl_config

    config_path = tmp_path / "dfl.yaml"
    payload = {
        "dataset": {
            "extracted_dir": "data/extracted",
            "index_path": "data/processed/image_index.jsonl",
            "train_test_manifest_path": "data/splits/train_test_manifest.json",
            "partition_manifest_path": "data/splits/dirichlet_clients_manifest.json",
            "image_size": [64, 64],
        },
        "training": {
            "num_agents": 10,
            "rounds": 1,
            "local_epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.01,
        },
        "dfl": {
            "encounters_per_round": 6,
        },
        "model": {
            "num_classes": 2,
            "input_channels": 3,
        },
        "artifacts": {
            "metrics_path": "artifacts/metrics/dfl/metrics.csv",
            "models_dir": "artifacts/models/dfl",
            "plot_path": "artifacts/plots/dfl/accuracy_vs_rounds.png",
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="encounters_per_round"):
        load_dfl_config(config_path)


def test_sample_random_pairings_without_replacement() -> None:
    from src.dfl.simulator import sample_random_pairings

    rng = random.Random(7)
    pairs, unpaired = sample_random_pairings(
        [f"client_{index:02d}" for index in range(10)],
        rng,
        max_pairs=5,
    )

    assert len(pairs) == 5
    assert unpaired == []

    flat_ids = [agent_id for pair in pairs for agent_id in pair]
    assert len(set(flat_ids)) == 10
    assert sorted(flat_ids) == [f"client_{index:02d}" for index in range(10)]
