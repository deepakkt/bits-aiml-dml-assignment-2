from __future__ import annotations

from pathlib import Path

from src.data.dataset import (
    build_dataset_index,
    build_train_test_split_manifest,
    load_dataset_split,
    locate_class_directories,
    read_index,
    read_split_manifest,
    write_index,
    write_split_manifest,
)


def _write_placeholder_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def _create_nested_dataset(extracted_dir: Path) -> None:
    pet_images_dir = extracted_dir / "cats-and-dogs-classification-dataset" / "PetImages"

    for index in range(4):
        _write_placeholder_image(pet_images_dir / "Cat" / f"cat_{index}.jpg")
    for index in range(4):
        _write_placeholder_image(pet_images_dir / "Dog" / f"dog_{index}.jpeg")

    # Non-image files must not appear in the dataset index.
    (pet_images_dir / "Cat" / "notes.txt").write_text("skip me", encoding="utf-8")


def test_locate_class_directories_handles_nested_layout(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "extracted"
    _create_nested_dataset(extracted_dir)

    class_dirs = locate_class_directories(extracted_dir)

    assert set(class_dirs) == {"cat", "dog"}
    assert class_dirs["cat"].name == "Cat"
    assert class_dirs["dog"].name == "Dog"


def test_index_and_split_manifest_are_deterministic(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "extracted"
    _create_nested_dataset(extracted_dir)

    index_records = build_dataset_index(extracted_dir)
    split_one = build_train_test_split_manifest(index_records, seed=7, train_ratio=0.75)
    split_two = build_train_test_split_manifest(index_records, seed=7, train_ratio=0.75)

    assert len(index_records) == 8
    assert split_one == split_two
    assert len(split_one["splits"]["train"]) == 6
    assert len(split_one["splits"]["test"]) == 2


def test_index_and_split_round_trip_loader(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "extracted"
    _create_nested_dataset(extracted_dir)

    index_records = build_dataset_index(extracted_dir)
    split_manifest = build_train_test_split_manifest(index_records, seed=42, train_ratio=0.75)

    index_path = tmp_path / "processed" / "image_index.jsonl"
    split_path = tmp_path / "splits" / "train_test_manifest.json"
    write_index(index_records, index_path)
    write_split_manifest(split_manifest, split_path)

    loaded_records = read_index(index_path)
    loaded_manifest = read_split_manifest(split_path)
    train_samples = load_dataset_split(
        extracted_dir=extracted_dir,
        index_path=index_path,
        split_manifest_path=split_path,
        split="train",
    )

    assert loaded_records == index_records
    assert loaded_manifest == split_manifest
    assert len(train_samples) == len(split_manifest["splits"]["train"])
    assert all(sample_path.exists() for sample_path, _ in train_samples)
