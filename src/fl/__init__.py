from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FedAvgConfig": ("src.fl.fedavg", "FedAvgConfig"),
    "load_fedavg_config": ("src.fl.fedavg", "load_fedavg_config"),
    "run_fedavg_experiment": ("src.fl.fedavg", "run_fedavg_experiment"),
    "weighted_average_state_dicts": ("src.fl.fedavg", "weighted_average_state_dicts"),
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
