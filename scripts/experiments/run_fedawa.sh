#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${1:-configs/fedawa.yaml}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "FedAWA config not found: $CONFIG_PATH" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Run ./scripts/dev/bootstrap_venv.sh first." >&2
  exit 1
fi

PARTITION_MANIFEST_PATH="$(
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"]["partition_manifest_path"])' \
    "$CONFIG_PATH"
)"

if [ ! -f "$PARTITION_MANIFEST_PATH" ]; then
  echo "Dirichlet partition manifest missing at $PARTITION_MANIFEST_PATH"
  echo "Generating deterministic partition manifest (10 clients, alpha=0.5)..."
  "$PYTHON_BIN" -m src.data.partition_dirichlet
fi

"$PYTHON_BIN" -m src.main fedawa --config "$CONFIG_PATH"
