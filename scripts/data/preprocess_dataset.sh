#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZIP_PATH="$ROOT_DIR/data/raw/cats-and-dogs-classification-dataset.zip"
EXTRACT_DIR="$ROOT_DIR/data/extracted"
PROCESSED_DIR="$ROOT_DIR/data/processed"
SPLITS_DIR="$ROOT_DIR/data/splits"
INDEX_PATH="$PROCESSED_DIR/image_index.jsonl"
SPLIT_MANIFEST_PATH="$SPLITS_DIR/train_test_manifest.json"

SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"

mkdir -p "$EXTRACT_DIR" "$PROCESSED_DIR" "$SPLITS_DIR"

if [ ! -s "$ZIP_PATH" ]; then
  echo "Dataset zip not found at: $ZIP_PATH" >&2
  echo "Place the dataset zip there or run scripts/data/get_dataset.sh first." >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip command not found. Please install unzip and rerun." >&2
  exit 1
fi

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No python interpreter found. Install python3.11 or create .venv first." >&2
  exit 1
fi

CHECKSUM="$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')"
EXTRACTION_MARKER="$EXTRACT_DIR/.extract_${CHECKSUM}.done"
PREPROCESS_MARKER="$PROCESSED_DIR/.preprocess_${CHECKSUM}.done"

if [ -f "$PREPROCESS_MARKER" ] && [ -f "$INDEX_PATH" ] && [ -f "$SPLIT_MANIFEST_PATH" ]; then
  echo "Preprocessing already completed for checksum $CHECKSUM; skipping."
  exit 0
fi

if [ ! -f "$EXTRACTION_MARKER" ]; then
  echo "Extracting dataset zip to $EXTRACT_DIR ..."
  unzip -oq "$ZIP_PATH" -d "$EXTRACT_DIR"
  touch "$EXTRACTION_MARKER"
else
  echo "Extraction marker found for checksum $CHECKSUM; reusing extracted files."
fi

echo "Building deterministic index and split manifest..."
"$PYTHON_BIN" -m src.data.preprocess \
  --extracted-dir "$EXTRACT_DIR" \
  --index-path "$INDEX_PATH" \
  --split-manifest-path "$SPLIT_MANIFEST_PATH" \
  --seed "$SPLIT_SEED" \
  --train-ratio "$TRAIN_RATIO"

touch "$PREPROCESS_MARKER"
echo "$CHECKSUM" > "$PROCESSED_DIR/current_dataset_checksum.txt"

echo "Preprocessing complete."
echo "Index: $INDEX_PATH"
echo "Split manifest: $SPLIT_MANIFEST_PATH"
