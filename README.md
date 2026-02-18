# Advanced Federated Learning Assignment

This repository contains an incremental implementation of the Advanced Federated Learning programming assignment.

## Current Status

- Implemented: Part 1 (repo bootstrap and developer ergonomics), Part 2 (dataset handling + preprocessing), Part 3 (Dirichlet non-IID partitioning), Part 4 (model + train/eval utilities + serialization)
- Pending: Parts 5-13

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

## Dirichlet Non-IID Client Partitioning (Part 3)

After preprocessing creates `data/processed/image_index.jsonl` and
`data/splits/train_test_manifest.json`, generate a 10-client non-IID split using
Dirichlet `alpha=0.5`:

```bash
.venv/bin/python -m src.data.partition_dirichlet
```

Outputs:

- Client partition manifest: `data/splits/dirichlet_clients_manifest.json`
- Distribution sanity table: `data/splits/dirichlet_clients_sanity.csv`

Optional flags:

- `--num-clients` (default: `10`)
- `--alpha` (default: `0.5`)
- `--seed` (default: `42`)
- `--source-split` (default: `train`)

## Model + Training Utilities (Part 4)

Part 4 adds reusable training primitives for upcoming FL experiments:

- CNN model: `src/models/simple_cnn.py`
- Tensor dataset adapters: `src/data/torch_dataset.py`
- Local train/eval loops: `src/common/train_eval.py`
- `.pkl` snapshot save/load: `src/common/serialization.py`
- CSV metrics logger: `src/common/metrics.py`

Example snapshot path convention:

- `artifacts/models/<experiment_name>/round_<n>.pkl`

The snapshot payload structure is:

```python
{
    "model_name": "...",
    "round": 0,
    "state_dict": {...},
    "config": {...},
    "metrics": {...},
    "timestamp": "...",
}
```

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
