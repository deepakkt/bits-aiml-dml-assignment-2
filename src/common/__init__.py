from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EpochMetrics": ("src.common.train_eval", "EpochMetrics"),
    "MetricsCSVLogger": ("src.common.metrics", "MetricsCSVLogger"),
    "append_metrics_row": ("src.common.metrics", "append_metrics_row"),
    "evaluate_model": ("src.common.train_eval", "evaluate_model"),
    "load_model_snapshot": ("src.common.serialization", "load_model_snapshot"),
    "load_snapshot_into_model": ("src.common.serialization", "load_snapshot_into_model"),
    "save_model_snapshot": ("src.common.serialization", "save_model_snapshot"),
    "train_local_epochs": ("src.common.train_eval", "train_local_epochs"),
    "train_one_epoch": ("src.common.train_eval", "train_one_epoch"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
