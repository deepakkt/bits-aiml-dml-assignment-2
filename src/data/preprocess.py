from __future__ import annotations

import argparse
from pathlib import Path

from src.data.dataset import (
    build_dataset_index,
    build_train_test_split_manifest,
    write_index,
    write_split_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess cats-vs-dogs dataset.")
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("data/extracted"),
        help="Directory that contains the extracted dataset.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/processed/image_index.jsonl"),
        help="Output JSONL path for image index records.",
    )
    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        default=Path("data/splits/train_test_manifest.json"),
        help="Output JSON path for deterministic split manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic split generation.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of samples assigned to the train split.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index_records = build_dataset_index(args.extracted_dir)
    split_manifest = build_train_test_split_manifest(
        index_records,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )

    write_index(index_records, args.index_path)
    write_split_manifest(split_manifest, args.split_manifest_path)

    train_size = len(split_manifest["splits"]["train"])
    test_size = len(split_manifest["splits"]["test"])
    print(
        "Preprocessed dataset with "
        f"{len(index_records)} samples (train={train_size}, test={test_size})."
    )
    print(f"Index written to: {args.index_path}")
    print(f"Split manifest written to: {args.split_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
