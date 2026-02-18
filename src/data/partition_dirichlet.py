from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.data.dataset import DatasetRecord, read_index, read_split_manifest, write_split_manifest


def _sample_dirichlet_proportions(
    *,
    num_clients: int,
    alpha: float,
    random_generator: random.Random,
) -> list[float]:
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    weights = [random_generator.gammavariate(alpha, 1.0) for _ in range(num_clients)]
    total_weight = sum(weights)
    if total_weight == 0:
        return [1.0 / float(num_clients)] * num_clients
    return [weight / total_weight for weight in weights]


def _allocate_counts(total_items: int, proportions: list[float]) -> list[int]:
    if total_items < 0:
        raise ValueError(f"total_items must be >= 0, got {total_items}")
    if not proportions:
        raise ValueError("proportions must not be empty")

    raw_counts = [total_items * proportion for proportion in proportions]
    counts = [int(raw_count) for raw_count in raw_counts]
    remainder = total_items - sum(counts)

    if remainder > 0:
        order = sorted(
            range(len(raw_counts)),
            key=lambda index: raw_counts[index] - counts[index],
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1

    return counts


def build_dirichlet_client_manifest(
    records: list[DatasetRecord],
    sample_ids: list[int],
    *,
    num_clients: int = 10,
    alpha: float = 0.5,
    seed: int = 42,
    source_split: str = "train",
) -> dict[str, object]:
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    records_by_id = {record.sample_id: record for record in records}
    ids_by_label: dict[str, list[int]] = defaultdict(list)

    for sample_id in sample_ids:
        if sample_id not in records_by_id:
            raise ValueError(f"Sample id {sample_id} is not present in index records.")
        ids_by_label[records_by_id[sample_id].label].append(sample_id)

    random_generator = random.Random(seed)
    clients: list[list[int]] = [[] for _ in range(num_clients)]

    for label in sorted(ids_by_label):
        label_ids = sorted(ids_by_label[label])
        random_generator.shuffle(label_ids)

        proportions = _sample_dirichlet_proportions(
            num_clients=num_clients,
            alpha=alpha,
            random_generator=random_generator,
        )
        counts = _allocate_counts(len(label_ids), proportions)

        start = 0
        for client_index, count in enumerate(counts):
            stop = start + count
            clients[client_index].extend(label_ids[start:stop])
            start = stop

    width = max(2, len(str(num_clients - 1)))
    client_section: dict[str, object] = {}
    overall_distribution = Counter(records_by_id[sample_id].label for sample_id in sample_ids)

    for client_index, assigned_ids in enumerate(clients):
        assigned_ids.sort()
        label_distribution = Counter(records_by_id[sample_id].label for sample_id in assigned_ids)
        client_id = f"client_{client_index:0{width}d}"
        client_section[client_id] = {
            "num_samples": len(assigned_ids),
            "class_distribution": dict(sorted(label_distribution.items())),
            "sample_ids": assigned_ids,
        }

    return {
        "schema_version": 1,
        "partition_method": "dirichlet",
        "source_split": source_split,
        "num_clients": num_clients,
        "num_samples": len(sample_ids),
        "partition_params": {
            "alpha": alpha,
            "seed": seed,
        },
        "overall_class_distribution": dict(sorted(overall_distribution.items())),
        "clients": client_section,
    }


def write_partition_sanity_table(
    partition_manifest: dict[str, object],
    output_path: Path,
) -> None:
    clients = partition_manifest["clients"]
    if not isinstance(clients, dict):
        raise ValueError("Partition manifest has invalid 'clients' section.")

    labels = sorted(
        {
            str(label)
            for client_data in clients.values()
            for label in client_data.get("class_distribution", {})
        }
    )
    fieldnames = ["client_id", "num_samples", *labels, *[f"{label}_fraction" for label in labels]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for client_id in sorted(clients):
            client_data = clients[client_id]
            num_samples = int(client_data.get("num_samples", 0))
            class_distribution = client_data.get("class_distribution", {})
            row: dict[str, object] = {
                "client_id": client_id,
                "num_samples": num_samples,
            }
            for label in labels:
                label_count = int(class_distribution.get(label, 0))
                row[label] = label_count
                row[f"{label}_fraction"] = (
                    0.0 if num_samples == 0 else round(label_count / float(num_samples), 6)
                )
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create non-IID client partitions using Dirichlet distribution."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/processed/image_index.jsonl"),
        help="Path to the JSONL dataset index.",
    )
    parser.add_argument(
        "--train-test-manifest-path",
        type=Path,
        default=Path("data/splits/train_test_manifest.json"),
        help="Path to train/test split manifest JSON.",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default="train",
        help="Split name from train/test manifest to partition across clients.",
    )
    parser.add_argument(
        "--output-manifest-path",
        type=Path,
        default=Path("data/splits/dirichlet_clients_manifest.json"),
        help="Output JSON path for client partition manifest.",
    )
    parser.add_argument(
        "--sanity-table-path",
        type=Path,
        default=Path("data/splits/dirichlet_clients_sanity.csv"),
        help="Output CSV path for per-client class-distribution sanity table.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of clients in the non-IID partition.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha concentration parameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible partitioning.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    records = read_index(args.index_path)
    train_test_manifest = read_split_manifest(args.train_test_manifest_path)

    split_section = train_test_manifest.get("splits", {})
    if args.source_split not in split_section:
        raise KeyError(f"Split '{args.source_split}' not found in {args.train_test_manifest_path}.")
    sample_ids = [int(sample_id) for sample_id in split_section[args.source_split]]

    partition_manifest = build_dirichlet_client_manifest(
        records,
        sample_ids,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        source_split=args.source_split,
    )
    write_split_manifest(partition_manifest, args.output_manifest_path)
    write_partition_sanity_table(partition_manifest, args.sanity_table_path)

    print(
        f"Created Dirichlet partition for {partition_manifest['num_samples']} samples "
        f"across {args.num_clients} clients (alpha={args.alpha}, seed={args.seed})."
    )
    print(f"Client partition manifest written to: {args.output_manifest_path}")
    print(f"Sanity table written to: {args.sanity_table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
