from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_simple_cnn_forward_shape() -> None:
    from src.models.simple_cnn import SimpleCNN

    model = SimpleCNN(num_classes=2)
    inputs = torch.randn(4, 3, 64, 64)

    logits = model(inputs)

    assert logits.shape == (4, 2)


def test_train_local_epochs_and_evaluate() -> None:
    from src.common.train_eval import evaluate_model, train_local_epochs
    from src.models.simple_cnn import SimpleCNN
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(11)
    inputs = torch.rand(24, 3, 64, 64)
    labels = (inputs.mean(dim=(1, 2, 3)) > 0.5).long()

    data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=8, shuffle=False)
    model = SimpleCNN(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    before_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    train_metrics = train_local_epochs(
        model,
        data_loader,
        optimizer,
        criterion,
        local_epochs=2,
        device="cpu",
    )
    eval_metrics = evaluate_model(model, data_loader, criterion, device="cpu")

    assert len(train_metrics) == 2
    assert all(metric.num_examples == 24 for metric in train_metrics)
    assert all(0.0 <= metric.accuracy <= 1.0 for metric in train_metrics)
    assert eval_metrics.num_examples == 24
    assert 0.0 <= eval_metrics.accuracy <= 1.0

    after_state = model.state_dict()
    assert any(not torch.equal(before_state[name], after_state[name]) for name in before_state)


def test_snapshot_save_and_load_round_trip(tmp_path: Path) -> None:
    from src.common.serialization import (
        load_model_snapshot,
        load_snapshot_into_model,
        save_model_snapshot,
    )
    from src.models.simple_cnn import SimpleCNN

    torch.manual_seed(7)
    model = SimpleCNN(num_classes=2)
    snapshot_path = tmp_path / "models" / "round_3.pkl"

    original_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    payload = save_model_snapshot(
        model,
        snapshot_path,
        model_name="simple_cnn",
        round_index=3,
        config={"lr": 0.05, "batch_size": 16},
        metrics={"accuracy": 0.75},
    )

    assert snapshot_path.exists()
    assert payload["model_name"] == "simple_cnn"
    assert payload["round"] == 3

    loaded_payload = load_model_snapshot(snapshot_path)
    assert loaded_payload["model_name"] == "simple_cnn"
    assert loaded_payload["round"] == 3

    restored_model = SimpleCNN(num_classes=2)
    load_snapshot_into_model(restored_model, snapshot_path)

    for name, tensor in original_state.items():
        assert torch.equal(tensor, restored_model.state_dict()[name])


def test_metrics_csv_logger_writes_rows_and_validates_schema(tmp_path: Path) -> None:
    from src.common.metrics import MetricsCSVLogger

    metrics_path = tmp_path / "metrics" / "metrics.csv"
    logger = MetricsCSVLogger(metrics_path)

    logger.log({"round": 1, "accuracy": 0.5, "loss": 1.2})
    logger.log({"round": 2, "accuracy": 0.65, "loss": 0.9})

    with metrics_path.open("r", encoding="utf-8", newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    assert len(rows) == 2
    assert rows[0]["round"] == "1"
    assert rows[1]["accuracy"] == "0.65"

    with pytest.raises(ValueError):
        logger.log({"round": 3, "accuracy": 0.7})
