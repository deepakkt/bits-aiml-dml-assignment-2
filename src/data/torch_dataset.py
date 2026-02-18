from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.data.dataset import load_dataset_split, read_index, read_split_manifest

LABEL_TO_INDEX: Final[dict[str, int]] = {"cat": 0, "dog": 1}

try:
    _RESAMPLING_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1 fallback.
    _RESAMPLING_BILINEAR = Image.BILINEAR


class ImageClassificationDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        samples: list[tuple[Path, str]],
        *,
        image_size: tuple[int, int] = (64, 64),
    ) -> None:
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self._samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label = self._samples[index]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            resized = rgb_image.resize(self.image_size, _RESAMPLING_BILINEAR)
            image_array = np.asarray(resized, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        return image_tensor, label_to_index(label)


def label_to_index(label: str) -> int:
    canonical = label.strip().lower()
    if canonical not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown label '{label}'. Expected one of: {sorted(LABEL_TO_INDEX)}")
    return LABEL_TO_INDEX[canonical]


def load_client_samples(
    *,
    extracted_dir: Path,
    index_path: Path,
    partition_manifest_path: Path,
    client_id: str,
) -> list[tuple[Path, str]]:
    records = read_index(index_path)
    partition_manifest = read_split_manifest(partition_manifest_path)

    clients = partition_manifest.get("clients", {})
    if client_id not in clients:
        raise KeyError(
            f"Client '{client_id}' not found in partition manifest at {partition_manifest_path}."
        )

    sample_ids = clients[client_id].get("sample_ids", [])
    records_by_id = {record.sample_id: record for record in records}

    resolved_samples: list[tuple[Path, str]] = []
    for sample_id in sample_ids:
        sample_id_int = int(sample_id)
        if sample_id_int not in records_by_id:
            raise ValueError(
                f"Sample id {sample_id_int} from client '{client_id}' missing in index."
            )
        record = records_by_id[sample_id_int]
        resolved_samples.append((extracted_dir / record.relative_path, record.label))

    return resolved_samples


def build_image_dataset_from_split(
    *,
    extracted_dir: Path,
    index_path: Path,
    split_manifest_path: Path,
    split: str,
    image_size: tuple[int, int] = (64, 64),
) -> ImageClassificationDataset:
    samples = load_dataset_split(
        extracted_dir=extracted_dir,
        index_path=index_path,
        split_manifest_path=split_manifest_path,
        split=split,
    )
    return ImageClassificationDataset(samples, image_size=image_size)


def build_image_dataset_from_client(
    *,
    extracted_dir: Path,
    index_path: Path,
    partition_manifest_path: Path,
    client_id: str,
    image_size: tuple[int, int] = (64, 64),
) -> ImageClassificationDataset:
    samples = load_client_samples(
        extracted_dir=extracted_dir,
        index_path=index_path,
        partition_manifest_path=partition_manifest_path,
        client_id=client_id,
    )
    return ImageClassificationDataset(samples, image_size=image_size)
