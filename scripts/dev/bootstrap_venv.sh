#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 not found. Please install Python 3.11 and retry." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  python3.11 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -m pip install --requirement requirements.txt; then
  echo "Failed to install dependencies from requirements.txt." >&2
  echo "Check internet connectivity or configure a reachable package index." >&2
  exit 1
fi

echo "Virtual environment ready at $ROOT_DIR/.venv"
