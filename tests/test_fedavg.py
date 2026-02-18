from __future__ import annotations

import json
from pathlib import Path


def test_load_fedavg_config_from_json_compatible_yaml(tmp_path: Path) -> None:
    from src.fl.fedavg import load_fedavg_config

    config_path = tmp_path / "fedavg.yaml"
    payload = {
        "experiment_name": "fedavg-test",
        "seed": 9,
        "dataset": {
            "extracted_dir": "data/extracted",
            "index_path": "data/processed/image_index.jsonl",
            "train_test_manifest_path": "data/splits/train_test_manifest.json",
            "partition_manifest_path": "data/splits/dirichlet_clients_manifest.json",
            "image_size": [64, 64],
        },
        "fedavg": {
            "num_clients": 10,
            "dirichlet_alpha": 0.5,
            "rounds": 2,
            "local_epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.01,
        },
        "model": {
            "num_classes": 2,
            "input_channels": 3,
        },
        "artifacts": {
            "metrics_path": "artifacts/metrics/fedavg/metrics.csv",
            "models_dir": "artifacts/models/fedavg",
            "plot_path": "artifacts/plots/fedavg/accuracy_vs_rounds.png",
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_fedavg_config(config_path)

    assert config.experiment_name == "fedavg-test"
    assert config.seed == 9
    assert config.num_clients == 10
    assert config.dirichlet_alpha == 0.5


def test_save_accuracy_line_plot_writes_png(tmp_path: Path) -> None:
    from src.plotting.line_plot import save_accuracy_line_plot

    output_path = tmp_path / "plots" / "accuracy.png"

    save_accuracy_line_plot(
        rounds=[1, 2, 3, 4],
        accuracies=[0.50, 0.60, 0.72, 0.80],
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
