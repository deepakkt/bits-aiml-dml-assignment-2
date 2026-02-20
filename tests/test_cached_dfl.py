from __future__ import annotations

import json
from pathlib import Path

import pytest


def _valid_cached_dfl_payload() -> dict[str, object]:
    return {
        "experiment_name": "cached-dfl-test",
        "seed": 99,
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
            "cache_size": 3,
            "tau_max": 5,
            "run_no_cache_baseline": False,
        },
        "model": {
            "num_classes": 2,
            "input_channels": 3,
        },
        "artifacts": {
            "cached_metrics_path": "artifacts/metrics/cached_dfl/metrics.csv",
            "cached_models_dir": "artifacts/models/cached_dfl",
            "cached_plot_path": "artifacts/plots/cached_dfl/accuracy_vs_rounds.png",
            "no_cache_metrics_path": "artifacts/metrics/dfl/metrics.csv",
            "no_cache_models_dir": "artifacts/models/dfl",
            "no_cache_plot_path": "artifacts/plots/dfl/accuracy_vs_rounds.png",
            "comparison_plot_path": "artifacts/plots/comparison/dfl_vs_cached_dfl_accuracy.png",
        },
    }


def test_load_cached_dfl_config_from_json_compatible_yaml(tmp_path: Path) -> None:
    from src.dfl.cached_simulator import load_cached_dfl_config

    config_path = tmp_path / "cached_dfl.yaml"
    payload = _valid_cached_dfl_payload()
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_cached_dfl_config(config_path)

    assert config.experiment_name == "cached-dfl-test"
    assert config.seed == 99
    assert config.cache_size == 3
    assert config.tau_max == 5
    assert config.run_no_cache_baseline is False


def test_load_cached_dfl_config_rejects_non_positive_cache_size(tmp_path: Path) -> None:
    from src.dfl.cached_simulator import load_cached_dfl_config

    config_path = tmp_path / "cached_dfl.yaml"
    payload = _valid_cached_dfl_payload()
    payload["dfl"] = {**payload["dfl"], "cache_size": 0}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cache_size"):
        load_cached_dfl_config(config_path)


def test_evict_stale_cache_entries_respects_tau_max() -> None:
    from src.dfl.cached_simulator import CachedModelEntry, evict_stale_cache_entries

    entries = [
        CachedModelEntry(source_agent_id="a", round_seen=2, state_dict={}),
        CachedModelEntry(source_agent_id="b", round_seen=6, state_dict={}),
        CachedModelEntry(source_agent_id="c", round_seen=9, state_dict={}),
    ]

    retained, evicted = evict_stale_cache_entries(entries, current_round=10, tau_max=3)

    assert [entry.source_agent_id for entry in retained] == ["c"]
    assert evicted == 2


def test_upsert_cache_entry_replaces_source_and_evicts_oldest() -> None:
    from src.dfl.cached_simulator import CachedModelEntry, upsert_cache_entry

    entries = [
        CachedModelEntry(source_agent_id="a", round_seen=1, state_dict={}),
        CachedModelEntry(source_agent_id="b", round_seen=2, state_dict={}),
        CachedModelEntry(source_agent_id="c", round_seen=3, state_dict={}),
    ]

    replaced, replaced_evictions = upsert_cache_entry(
        entries,
        new_entry=CachedModelEntry(source_agent_id="b", round_seen=4, state_dict={}),
        cache_size=3,
    )

    assert replaced_evictions == 0
    assert [entry.source_agent_id for entry in replaced] == ["b", "c", "a"]

    retained, capacity_evictions = upsert_cache_entry(
        replaced,
        new_entry=CachedModelEntry(source_agent_id="d", round_seen=5, state_dict={}),
        cache_size=3,
    )

    assert capacity_evictions == 1
    assert [entry.source_agent_id for entry in retained] == ["d", "b", "c"]
