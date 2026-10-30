#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${CONFIG_FILE:-$ROOT_DIR/config/local.env}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE:-$ROOT_DIR/config/local.env}"
fi

GRID_ROOT="${GRID_ROOT:-$ROOT_DIR/third_party/GRID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output}"
DATASET="${DATASET:-beauty}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_DEVICES="${NUM_DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAINER_STRATEGY="${TRAINER_STRATEGY:-auto}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "$DATASET" != "beauty" && "$DATASET" != "toys" ]]; then
  echo "DATASET must be 'beauty' or 'toys'" >&2
  exit 2
fi
if [[ ! -d "$GRID_ROOT/src" ]]; then
  echo "GRID is missing. Run: bash scripts/bootstrap_grid.sh" >&2
  exit 2
fi
if [[ -z "${AMAZON_DATA_ROOT:-}" ]]; then
  echo "Set AMAZON_DATA_ROOT in config/local.env or the environment" >&2
  exit 2
fi

DATA_DIR="$AMAZON_DATA_ROOT/$DATASET"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export ROOT_DIR GRID_ROOT PYTHON_BIN OUTPUT_ROOT DATASET DATA_DIR LOG_ROOT
export CUDA_VISIBLE_DEVICES
export MASTER_PORT
export PYTHONPATH="$GRID_ROOT:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
