from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_load_fedawa_config_from_json_compatible_yaml(tmp_path: Path) -> None:
    from src.fl.fedawa import load_fedawa_config

    config_path = tmp_path / "fedawa.yaml"
    payload = {
        "experiment_name": "fedawa-test",
        "seed": 13,
        "dataset": {
            "extracted_dir": "data/extracted",
            "index_path": "data/processed/image_index.jsonl",
            "train_test_manifest_path": "data/splits/train_test_manifest.json",
            "partition_manifest_path": "data/splits/dirichlet_clients_manifest.json",
            "image_size": [64, 64],
        },
        "training": {
            "num_clients": 10,
            "dirichlet_alpha": 0.5,
            "rounds": 3,
            "local_epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.01,
        },
        "model": {
            "num_classes": 2,
            "input_channels": 3,
        },
        "fedawa": {
            "alignment_epsilon": 1e-9,
            "negative_alignment_mode": "clip_to_zero",
            "run_fedavg_baseline": False,
        },
        "artifacts": {
            "fedawa_metrics_path": "artifacts/metrics/fedawa/metrics.csv",
            "fedawa_models_dir": "artifacts/models/fedawa",
            "fedawa_plot_path": "artifacts/plots/fedawa/accuracy_vs_rounds.png",
            "fedavg_metrics_path": "artifacts/metrics/fedavg/metrics.csv",
            "fedavg_models_dir": "artifacts/models/fedavg",
            "fedavg_plot_path": "artifacts/plots/fedavg/accuracy_vs_rounds.png",
            "comparison_plot_path": "artifacts/plots/comparison/fedavg_vs_fedawa_accuracy.png",
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_fedawa_config(config_path)

    assert config.experiment_name == "fedawa-test"
    assert config.seed == 13
    assert config.num_clients == 10
    assert config.alignment_epsilon == pytest.approx(1e-9)
    assert config.run_fedavg_baseline is False


def test_compute_fedawa_client_weights_clips_negative_alignments() -> None:
    from src.fl.fedawa import compute_fedawa_client_weights

    weights = compute_fedawa_client_weights(
        cosine_alignments=[0.8, -0.4, 0.2],
        client_num_examples=[10, 10, 20],
        negative_alignment_mode="clip_to_zero",
    )

    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.0)
    assert weights[0] == pytest.approx(8.0 / 12.0)
    assert weights[2] == pytest.approx(4.0 / 12.0)


def test_compute_fedawa_client_weights_falls_back_to_dataset_size_when_all_non_positive() -> None:
    from src.fl.fedawa import compute_fedawa_client_weights

    weights = compute_fedawa_client_weights(
        cosine_alignments=[-0.2, 0.0, -0.1],
        client_num_examples=[3, 1, 2],
        negative_alignment_mode="clip_to_zero",
    )

    assert weights == pytest.approx([0.5, 1.0 / 6.0, 1.0 / 3.0])


def test_save_multi_accuracy_line_plot_writes_png(tmp_path: Path) -> None:
    from src.plotting.line_plot import save_multi_accuracy_line_plot

    output_path = tmp_path / "plots" / "fedavg_vs_fedawa.png"
    save_multi_accuracy_line_plot(
        series=[
            ("FedAvg", [1, 2, 3], [0.52, 0.58, 0.63]),
            ("FedAWA", [1, 2, 3], [0.55, 0.62, 0.69]),
        ],
        output_path=output_path,
        title="FedAvg vs FedAWA",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
