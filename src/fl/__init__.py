from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "FedAvgConfig": ("src.fl.fedavg", "FedAvgConfig"),
    "FedAWAConfig": ("src.fl.fedawa", "FedAWAConfig"),
    "load_fedavg_config": ("src.fl.fedavg", "load_fedavg_config"),
    "load_fedawa_config": ("src.fl.fedawa", "load_fedawa_config"),
    "run_fedawa_experiment": ("src.fl.fedawa", "run_fedawa_experiment"),
    "run_fedavg_experiment": ("src.fl.fedavg", "run_fedavg_experiment"),
    "compute_fedawa_client_weights": ("src.fl.fedawa", "compute_fedawa_client_weights"),
    "average_state_dicts_with_weights": ("src.fl.fedawa", "average_state_dicts_with_weights"),
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
