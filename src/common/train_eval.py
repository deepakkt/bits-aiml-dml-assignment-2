from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    num_examples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "num_examples": self.num_examples,
        }


def _build_metrics(*, loss_sum: float, correct: int, num_examples: int) -> EpochMetrics:
    if num_examples == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0, num_examples=0)
    return EpochMetrics(
        loss=loss_sum / float(num_examples),
        accuracy=correct / float(num_examples),
        num_examples=num_examples,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> EpochMetrics:
    model.to(device)
    model.train()

    loss_sum = 0.0
    correct = 0
    num_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.size(0))
        loss_sum += float(loss.item()) * batch_size
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        num_examples += batch_size

    return _build_metrics(loss_sum=loss_sum, correct=correct, num_examples=num_examples)


def train_local_epochs(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    local_epochs: int,
    device: str | torch.device = "cpu",
) -> list[EpochMetrics]:
    if local_epochs <= 0:
        raise ValueError(f"local_epochs must be > 0, got {local_epochs}")

    epoch_metrics: list[EpochMetrics] = []
    for _ in range(local_epochs):
        metrics = train_one_epoch(
            model,
            data_loader,
            optimizer,
            criterion,
            device=device,
        )
        epoch_metrics.append(metrics)
    return epoch_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> EpochMetrics:
    model.to(device)
    model.eval()

    loss_sum = 0.0
    correct = 0
    num_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        batch_size = int(labels.size(0))
        loss_sum += float(loss.item()) * batch_size
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        num_examples += batch_size

    return _build_metrics(loss_sum=loss_sum, correct=correct, num_examples=num_examples)
