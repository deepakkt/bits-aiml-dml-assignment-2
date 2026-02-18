# Advanced Federated Learning Assignment

This repository contains an incremental implementation of the Advanced Federated Learning programming assignment.

## Current Status

- Implemented: Part 1 (repo bootstrap and developer ergonomics)
- Pending: Parts 2-13

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

## Dataset Placement

If you already have the dataset zip, keep it at:

- `data/raw/cats-and-dogs-classification-dataset.zip`

Dataset acquisition and preprocessing scripts are added in later parts.

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
