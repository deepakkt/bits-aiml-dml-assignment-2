from __future__ import annotations

import csv
from pathlib import Path

from src.data.dataset import DatasetRecord
from src.data.partition_dirichlet import (
    build_dirichlet_client_manifest,
    write_partition_sanity_table,
)


def _build_records(num_cat: int, num_dog: int) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    sample_id = 0
    for index in range(num_cat):
        records.append(
            DatasetRecord(sample_id=sample_id, label="cat", relative_path=f"Cat/cat_{index}.jpg")
        )
        sample_id += 1
    for index in range(num_dog):
        records.append(
            DatasetRecord(sample_id=sample_id, label="dog", relative_path=f"Dog/dog_{index}.jpg")
        )
        sample_id += 1
    return records


def test_dirichlet_partition_is_deterministic() -> None:
    records = _build_records(num_cat=10, num_dog=10)
    sample_ids = [record.sample_id for record in records]

    first_manifest = build_dirichlet_client_manifest(
        records,
        sample_ids,
        num_clients=4,
        alpha=0.5,
        seed=9,
    )
    second_manifest = build_dirichlet_client_manifest(
        records,
        sample_ids,
        num_clients=4,
        alpha=0.5,
        seed=9,
    )

    assert first_manifest == second_manifest


def test_dirichlet_partition_assigns_each_sample_once() -> None:
    records = _build_records(num_cat=6, num_dog=6)
    sample_ids = [record.sample_id for record in records]

    manifest = build_dirichlet_client_manifest(
        records,
        sample_ids,
        num_clients=5,
        alpha=0.5,
        seed=42,
    )

    clients = manifest["clients"]
    all_assigned_ids = []
    for client_data in clients.values():
        all_assigned_ids.extend(client_data["sample_ids"])

    assert sorted(all_assigned_ids) == sorted(sample_ids)
    assert len(all_assigned_ids) == len(set(all_assigned_ids))


def test_sanity_table_writer_creates_expected_columns(tmp_path: Path) -> None:
    records = _build_records(num_cat=4, num_dog=4)
    sample_ids = [record.sample_id for record in records]
    manifest = build_dirichlet_client_manifest(
        records,
        sample_ids,
        num_clients=3,
        alpha=0.5,
        seed=4,
    )

    output_path = tmp_path / "dirichlet_sanity.csv"
    write_partition_sanity_table(manifest, output_path)

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows
    assert set(rows[0]) >= {
        "client_id",
        "num_samples",
        "cat",
        "dog",
        "cat_fraction",
        "dog_fraction",
    }
