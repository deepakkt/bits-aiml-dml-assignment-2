from __future__ import annotations

import pickle
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

REQUIRED_SNAPSHOT_KEYS = frozenset(
    {
        "model_name",
        "round",
        "state_dict",
        "config",
        "metrics",
        "timestamp",
    }
)


def _to_cpu_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}


def save_model_snapshot(
    model: nn.Module,
    output_path: Path,
    *,
    model_name: str,
    round_index: int,
    config: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if round_index < 0:
        raise ValueError(f"round_index must be >= 0, got {round_index}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": model_name,
        "round": round_index,
        "state_dict": _to_cpu_state_dict(model.state_dict()),
        "config": dict(config or {}),
        "metrics": dict(metrics or {}),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    with output_path.open("wb") as output_file:
        pickle.dump(payload, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    return payload


def load_model_snapshot(
    snapshot_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    with snapshot_path.open("rb") as snapshot_file:
        payload = pickle.load(snapshot_file)  # noqa: S301

    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot at {snapshot_path} must contain a dictionary payload.")

    missing_keys = sorted(REQUIRED_SNAPSHOT_KEYS.difference(payload))
    if missing_keys:
        raise ValueError(
            f"Snapshot at {snapshot_path} is missing required keys: {', '.join(missing_keys)}"
        )

    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"Snapshot at {snapshot_path} has invalid 'state_dict' section.")

    remapped_state_dict: dict[str, Tensor] = {}
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"State dict entry '{name}' is not a tensor.")
        remapped_state_dict[str(name)] = tensor.to(map_location)
    payload["state_dict"] = remapped_state_dict

    return payload


def load_snapshot_into_model(
    model: nn.Module,
    snapshot_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = load_model_snapshot(snapshot_path, map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return payload
