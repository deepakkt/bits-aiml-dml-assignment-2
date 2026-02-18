# Implementation State

## Completed Parts

- [x] Part 1 — Repo bootstrap & developer ergonomics
- [x] Part 2 — Dataset handling + preprocessing
- [ ] Part 3 — Dirichlet non-IID partitioning
- [ ] Part 4 — Model + train/eval + serialization
- [ ] Part 5 — FedAvg end-to-end
- [ ] Part 6 — FedAWA + comparison
- [ ] Part 7 — DFL baseline (no caching)
- [ ] Part 8 — Cached-DFL + comparison
- [ ] Part 9 — Report generation (PDF)
- [ ] Part 10 — CI + smoke tests
- [ ] Part 11 — Docker packaging
- [ ] Part 12 — Kubernetes + Argo CD + monitoring scripts
- [ ] Part 13 — Final polish

## Key Decisions

- Runtime/environment:
  - Python runtime pinned to `3.11` via `scripts/dev/bootstrap_venv.sh`.
  - Bootstrap remains idempotent (`.venv` created only if missing).
  - `pytest` config now includes `pythonpath = ["."]` so `src.*` imports resolve during tests.
- Pinned dependencies:
  - `black==24.10.0`
  - `kaggle==1.6.17`
  - `pytest==8.3.3`
  - `ruff==0.6.9`
- Data handling (Part 2):
  - Dataset zip source of truth path: `data/raw/cats-and-dogs-classification-dataset.zip`.
  - `scripts/data/get_dataset.sh` behavior:
    - reuses pre-placed zip (no re-download)
    - attempts Kaggle fallback only when credentials are detected
    - fails clearly when credentials are missing
    - supports optional `KAGGLE_DATASET`/`KAGGLE_FILE` overrides
  - `scripts/data/preprocess_dataset.sh` behavior:
    - checksum markers:
      - extraction marker: `data/extracted/.extract_<sha256>.done`
      - preprocess marker: `data/processed/.preprocess_<sha256>.done`
    - deterministic split defaults:
      - `SPLIT_SEED=42`
      - `TRAIN_RATIO=0.8`
  - Dataset utilities in `src/data/dataset.py`:
    - robust nested layout discovery (`Cat(s)` + `Dog(s)` directories up to 2 levels deep)
    - image index format (`data/processed/image_index.jsonl`):
      - per line: `{id, label, path}`
    - split manifest format (`data/splits/train_test_manifest.json`):
      - metadata + `splits.train` and `splits.test` sample IDs
- Git-LFS rule retained:
  - `data/raw/*.zip filter=lfs diff=lfs merge=lfs -text`

## File Paths Introduced/Updated in Part 2

- `configs/dataset.yaml`
- `pyproject.toml` (pytest path fix for stable test discovery)
- `scripts/data/get_dataset.sh`
- `scripts/data/preprocess_dataset.sh`
- `requirements.txt`
- `README.md`
- `src/data/__init__.py`
- `src/data/dataset.py`
- `src/data/preprocess.py`
- `state.md`
- `tests/test_dataset.py`

## How To Run

```bash
./scripts/dev/bootstrap_venv.sh
./scripts/data/get_dataset.sh
./scripts/data/preprocess_dataset.sh
make lint
make format-check
make test
```

## Gotchas

- `python3.11` must be available on `PATH`.
- `scripts/data/get_dataset.sh` only performs Kaggle fallback if credentials are available.
- If Kaggle dataset has multiple zip files, set `KAGGLE_FILE` explicitly.
- `preprocess_dataset.sh` relies on `unzip` and checksum utility `shasum`.
- Corrupt image files (if any) are still indexed by extension; filtering invalid files can be added later if needed.

## Next Part Handoff (Part 3)

- Implement Dirichlet non-IID partitioning with `alpha=0.5` for `10` clients.
- Persist partition manifests under `data/splits/` with per-client sample IDs.
- Add partition sanity output (table and/or plot) for quick validation.
