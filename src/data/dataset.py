from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
CLASS_NAME_ALIASES = {
    "cat": "cat",
    "cats": "cat",
    "dog": "dog",
    "dogs": "dog",
}
REQUIRED_CLASSES = frozenset({"cat", "dog"})


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: int
    label: str
    relative_path: str


def _canonical_label(name: str) -> str | None:
    return CLASS_NAME_ALIASES.get(name.strip().lower())


def _candidate_roots(extracted_dir: Path) -> list[Path]:
    level_one_dirs = sorted([path for path in extracted_dir.iterdir() if path.is_dir()])
    candidates = [extracted_dir, *level_one_dirs]
    for level_one_dir in level_one_dirs:
        level_two_dirs = sorted([path for path in level_one_dir.iterdir() if path.is_dir()])
        candidates.extend(level_two_dirs)
    return candidates


def locate_class_directories(extracted_dir: Path) -> dict[str, Path]:
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Extracted dataset directory does not exist: {extracted_dir}")

    for candidate_root in _candidate_roots(extracted_dir):
        class_dirs: dict[str, Path] = {}
        for child_dir in sorted([path for path in candidate_root.iterdir() if path.is_dir()]):
            canonical = _canonical_label(child_dir.name)
            if canonical is not None and canonical not in class_dirs:
                class_dirs[canonical] = child_dir

        if REQUIRED_CLASSES.issubset(class_dirs):
            return class_dirs

    raise FileNotFoundError(
        "Could not locate class directories for cat/dog dataset under "
        f"{extracted_dir}. Expected folders named Cat(s) and Dog(s)."
    )


def _iter_image_paths(directory: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.strip().lower() in IMAGE_SUFFIXES
        ]
    )


def build_dataset_index(extracted_dir: Path) -> list[DatasetRecord]:
    class_dirs = locate_class_directories(extracted_dir)
    indexed_rows: list[tuple[str, str]] = []

    for label, class_dir in sorted(class_dirs.items()):
        for image_path in _iter_image_paths(class_dir):
            indexed_rows.append((image_path.relative_to(extracted_dir).as_posix(), label))

    indexed_rows.sort(key=lambda row: row[0])
    if not indexed_rows:
        raise ValueError(f"No image files found under extracted directory: {extracted_dir}")

    return [
        DatasetRecord(sample_id=sample_id, label=label, relative_path=relative_path)
        for sample_id, (relative_path, label) in enumerate(indexed_rows)
    ]


def build_train_test_split_manifest(
    records: Iterable[DatasetRecord],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> dict[str, object]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    record_list = list(records)
    if not record_list:
        raise ValueError("Cannot create split manifest from an empty index.")

    ids_by_label: dict[str, list[int]] = defaultdict(list)
    for record in record_list:
        ids_by_label[record.label].append(record.sample_id)

    random_generator = random.Random(seed)
    train_ids: list[int] = []
    test_ids: list[int] = []

    for label in sorted(ids_by_label):
        label_ids = sorted(ids_by_label[label])
        random_generator.shuffle(label_ids)

        if len(label_ids) == 1:
            split_index = 1
        else:
            split_index = int(len(label_ids) * train_ratio)
            split_index = max(1, min(len(label_ids) - 1, split_index))

        train_ids.extend(label_ids[:split_index])
        test_ids.extend(label_ids[split_index:])

    train_ids.sort()
    test_ids.sort()

    return {
        "schema_version": 1,
        "seed": seed,
        "train_ratio": train_ratio,
        "num_samples": len(record_list),
        "class_distribution": {label: len(ids) for label, ids in sorted(ids_by_label.items())},
        "splits": {
            "train": train_ids,
            "test": test_ids,
        },
    }


def write_index(records: Iterable[DatasetRecord], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as index_file:
        for record in records:
            index_file.write(
                json.dumps(
                    {"id": record.sample_id, "label": record.label, "path": record.relative_path},
                    sort_keys=True,
                )
            )
            index_file.write("\n")


def read_index(index_path: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with index_path.open("r", encoding="utf-8") as index_file:
        for line in index_file:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            records.append(
                DatasetRecord(
                    sample_id=int(row["id"]),
                    label=str(row["label"]),
                    relative_path=str(row["path"]),
                )
            )
    return records


def write_split_manifest(manifest: dict[str, object], split_manifest_path: Path) -> None:
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with split_manifest_path.open("w", encoding="utf-8") as split_file:
        json.dump(manifest, split_file, indent=2, sort_keys=True)
        split_file.write("\n")


def read_split_manifest(split_manifest_path: Path) -> dict[str, object]:
    with split_manifest_path.open("r", encoding="utf-8") as split_file:
        return json.load(split_file)


def load_dataset_split(
    *,
    extracted_dir: Path,
    index_path: Path,
    split_manifest_path: Path,
    split: str,
) -> list[tuple[Path, str]]:
    records = read_index(index_path)
    manifest = read_split_manifest(split_manifest_path)

    split_section = manifest.get("splits", {})
    if split not in split_section:
        raise KeyError(f"Split '{split}' not found in split manifest at {split_manifest_path}.")

    split_ids = sorted(int(sample_id) for sample_id in split_section[split])
    records_by_id = {record.sample_id: record for record in records}
    resolved_samples: list[tuple[Path, str]] = []

    for sample_id in split_ids:
        if sample_id not in records_by_id:
            raise ValueError(
                f"Sample id {sample_id} is present in split manifest but not in index."
            )
        record = records_by_id[sample_id]
        resolved_samples.append((extracted_dir / record.relative_path, record.label))

    return resolved_samples
