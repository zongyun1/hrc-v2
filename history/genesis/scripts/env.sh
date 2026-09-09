#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GENESIS_HR_LATEST_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export GENESIS_LATEST_VENV="$GENESIS_HR_LATEST_ROOT/.venv-genesis-latest"

if [[ ! -x "$GENESIS_LATEST_VENV/bin/python" ]]; then
  echo "Missing venv: $GENESIS_LATEST_VENV" >&2
  echo "Rebuild with: python3 -m venv .venv-genesis-latest && .venv-genesis-latest/bin/python -m pip install -r requirements-genesis-latest.txt" >&2
  return 1 2>/dev/null || exit 1
fi

source "$GENESIS_LATEST_VENV/bin/activate"
export PYTHONPATH="$GENESIS_HR_LATEST_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GENESIS_BACKEND="${GENESIS_BACKEND:-gpu}"
