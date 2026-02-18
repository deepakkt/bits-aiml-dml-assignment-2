#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
ZIP_PATH="$RAW_DIR/cats-and-dogs-classification-dataset.zip"
TMP_DIR="$RAW_DIR/.kaggle_tmp_download"

# Expected Kaggle source can be overridden by the user.
KAGGLE_DATASET="${KAGGLE_DATASET:-tongpython/cat-and-dog}"
KAGGLE_FILE="${KAGGLE_FILE:-}"

mkdir -p "$RAW_DIR"

if [ -s "$ZIP_PATH" ]; then
  echo "Reusing existing dataset zip at: $ZIP_PATH"
  exit 0
fi

has_kaggle_credentials=0
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  has_kaggle_credentials=1
elif [ -f "${HOME}/.kaggle/kaggle.json" ]; then
  has_kaggle_credentials=1
fi

if [ "$has_kaggle_credentials" -eq 0 ]; then
  echo "Dataset zip not found at: $ZIP_PATH" >&2
  echo "Kaggle credentials are required for download fallback." >&2
  echo "Provide either:" >&2
  echo "  1) KAGGLE_USERNAME + KAGGLE_KEY environment variables, or" >&2
  echo "  2) ~/.kaggle/kaggle.json" >&2
  echo "Then rerun this script." >&2
  exit 1
fi

if command -v kaggle >/dev/null 2>&1; then
  KAGGLE_BIN="$(command -v kaggle)"
elif [ -x "$ROOT_DIR/.venv/bin/kaggle" ]; then
  KAGGLE_BIN="$ROOT_DIR/.venv/bin/kaggle"
else
  echo "Kaggle CLI not found. Install it and retry (e.g. pip install kaggle==1.6.17)." >&2
  exit 1
fi

mkdir -p "$TMP_DIR"
echo "Downloading dataset from Kaggle dataset '$KAGGLE_DATASET'..."

if [ -n "$KAGGLE_FILE" ]; then
  "$KAGGLE_BIN" datasets download -d "$KAGGLE_DATASET" -f "$KAGGLE_FILE" -p "$TMP_DIR"
  DOWNLOADED_ZIP="$TMP_DIR/$KAGGLE_FILE"
else
  "$KAGGLE_BIN" datasets download -d "$KAGGLE_DATASET" -p "$TMP_DIR"
  mapfile -t FOUND_ZIPS < <(find "$TMP_DIR" -maxdepth 1 -type f -name '*.zip' | sort)
  if [ "${#FOUND_ZIPS[@]}" -ne 1 ]; then
    echo "Expected exactly one downloaded zip in $TMP_DIR but found ${#FOUND_ZIPS[@]}." >&2
    echo "Set KAGGLE_FILE to the expected zip name and rerun." >&2
    exit 1
  fi
  DOWNLOADED_ZIP="${FOUND_ZIPS[0]}"
fi

if [ ! -s "$DOWNLOADED_ZIP" ]; then
  echo "Download failed: zip file was not created: $DOWNLOADED_ZIP" >&2
  exit 1
fi

mv -f "$DOWNLOADED_ZIP" "$ZIP_PATH"
rm -rf "$TMP_DIR"

echo "Dataset ready at: $ZIP_PATH"
