# Implementation State

## Completed Parts

- [x] Part 1 — Repo bootstrap & developer ergonomics
- [ ] Part 2 — Dataset handling + preprocessing
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

## Key Decisions (Part 1)

- Python runtime pinned to `3.11` via `scripts/dev/bootstrap_venv.sh`.
- Added a deterministic, idempotent bootstrap script that:
  - checks for `python3.11`
  - creates `.venv` only if missing
  - installs pinned dependencies from `requirements.txt`
- Pinned tooling versions:
  - `black==24.10.0`
  - `pytest==8.3.3`
  - `ruff==0.6.9`
- Added Git-LFS tracking rule for dataset zip in `.gitattributes`:
  - `data/raw/*.zip filter=lfs diff=lfs merge=lfs -text`
- Introduced baseline project scaffolding for assignment-aligned folders.
- Added minimal CLI entrypoint (`src/main.py`) and a smoke unit test (`tests/test_main.py`).

## File Paths Introduced/Updated in Part 1

- `.gitignore`
- `.gitattributes`
- `requirements.txt`
- `pyproject.toml`
- `README.md`
- `state.md`
- `Makefile`
- `scripts/dev/bootstrap_venv.sh`
- `src/main.py`
- `src/__init__.py`
- `tests/test_main.py`
- scaffold placeholder files (`.gitkeep`) in empty directories

## How To Run

```bash
./scripts/dev/bootstrap_venv.sh
make lint
make format-check
make test
make run
```

## Gotchas

- `python3.11` must be available on `PATH`.
- Bootstrapping needs access to a Python package index unless dependencies are pre-cached.
- This part intentionally does not include dataset download/preprocessing logic yet.

## Next Part Handoff (Part 2)

- Implement `scripts/data/get_dataset.sh` with:
  - pre-placed zip reuse behavior
  - optional Kaggle download fallback
  - clear credential-missing failure mode
- Implement `scripts/data/preprocess_dataset.sh` with idempotent extraction/indexing and marker files.
- Add Python-side dataset indexing/loader utilities under `src/data/`.
