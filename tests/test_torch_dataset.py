from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("torch")


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((32, 32, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _create_dataset(extracted_dir: Path) -> None:
    root = extracted_dir / "cats-and-dogs-classification-dataset" / "PetImages"

    for index in range(4):
        _write_image(root / "Cat" / f"cat_{index}.jpg", value=20 + index)
    for index in range(4):
        _write_image(root / "Dog" / f"dog_{index}.jpg", value=200 - index)


def test_build_image_dataset_from_split_and_client(tmp_path: Path) -> None:
    from src.data.dataset import (
        build_dataset_index,
        build_train_test_split_manifest,
        write_index,
        write_split_manifest,
    )
    from src.data.partition_dirichlet import build_dirichlet_client_manifest
    from src.data.torch_dataset import (
        build_image_dataset_from_client,
        build_image_dataset_from_split,
    )

    extracted_dir = tmp_path / "extracted"
    _create_dataset(extracted_dir)

    index_records = build_dataset_index(extracted_dir)
    split_manifest = build_train_test_split_manifest(index_records, seed=13, train_ratio=0.75)

    index_path = tmp_path / "processed" / "image_index.jsonl"
    split_manifest_path = tmp_path / "splits" / "train_test_manifest.json"
    partition_manifest_path = tmp_path / "splits" / "dirichlet_clients_manifest.json"

    write_index(index_records, index_path)
    write_split_manifest(split_manifest, split_manifest_path)

    sample_ids = [int(sample_id) for sample_id in split_manifest["splits"]["train"]]
    partition_manifest = build_dirichlet_client_manifest(
        index_records,
        sample_ids,
        num_clients=3,
        alpha=0.5,
        seed=17,
    )
    write_split_manifest(partition_manifest, partition_manifest_path)

    split_dataset = build_image_dataset_from_split(
        extracted_dir=extracted_dir,
        index_path=index_path,
        split_manifest_path=split_manifest_path,
        split="train",
        image_size=(64, 64),
    )

    assert len(split_dataset) == len(split_manifest["splits"]["train"])
    first_image, first_label = split_dataset[0]
    assert tuple(first_image.shape) == (3, 64, 64)
    assert float(first_image.min()) >= 0.0
    assert float(first_image.max()) <= 1.0
    assert first_label in {0, 1}

    clients = partition_manifest["clients"]
    non_empty_client = next(
        client_id for client_id, payload in clients.items() if payload["num_samples"] > 0
    )
    client_dataset = build_image_dataset_from_client(
        extracted_dir=extracted_dir,
        index_path=index_path,
        partition_manifest_path=partition_manifest_path,
        client_id=non_empty_client,
        image_size=(64, 64),
    )

    assert len(client_dataset) == int(clients[non_empty_client]["num_samples"])
