# Advanced Federated Learning Assignment

This repository contains an incremental implementation of the Advanced Federated Learning programming assignment.

## Current Status

- Implemented: Part 1 (repo bootstrap and developer ergonomics), Part 2 (dataset handling + preprocessing)
- Pending: Parts 3-13

## Quick Start

1. Ensure `python3.11` is installed.
2. Create/update virtual environment and install dependencies:

```bash
./scripts/dev/bootstrap_venv.sh
```

3. Run developer checks:

```bash
make lint
make format-check
make test
```

4. Run the placeholder CLI:

```bash
make run
```

## Dataset Pipeline (Part 2)

### 1) Acquire zip (idempotent)

Preferred: pre-place the zip here:

- `data/raw/cats-and-dogs-classification-dataset.zip`

Then run:

```bash
./scripts/data/get_dataset.sh
```

If the zip is already present, the script reuses it and exits.
If missing, it will attempt Kaggle download only when credentials are configured
(`KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`).

Optional download overrides:

- `KAGGLE_DATASET` (default: `tongpython/cat-and-dog`)
- `KAGGLE_FILE` (required when the dataset has multiple zip files)

### 2) Extract and preprocess (idempotent)

```bash
./scripts/data/preprocess_dataset.sh
```

This script:

- extracts the zip into `data/extracted/` with checksum marker files
- builds image index at `data/processed/image_index.jsonl`
- generates deterministic train/test split manifest at `data/splits/train_test_manifest.json`

Optional env overrides:

- `SPLIT_SEED` (default: `42`)
- `TRAIN_RATIO` (default: `0.8`)

## Repository Layout

```text
.github/workflows/
configs/
data/
  raw/
  extracted/
  processed/
  splits/
src/
  common/
  data/
  models/
  fl/
  dfl/
  plotting/
scripts/
  dev/
  data/
  experiments/
  k8s/
docker/
k8s/
  monitoring/
report/
tests/
```
