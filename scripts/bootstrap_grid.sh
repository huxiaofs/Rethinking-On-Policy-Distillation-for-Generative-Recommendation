#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_ROOT="${GRID_ROOT:-$ROOT_DIR/third_party/GRID}"
GRID_COMMIT="2fe3475b2d369580234093f35d52b1a2f54d0472"

if [[ ! -d "$GRID_ROOT/.git" ]]; then
  git clone https://github.com/snap-research/GRID.git "$GRID_ROOT"
fi
git -C "$GRID_ROOT" fetch origin "$GRID_COMMIT"
git -C "$GRID_ROOT" checkout --detach "$GRID_COMMIT"
touch "$GRID_ROOT/.project-root"

# Two experiment configs used by the public reproduction.
cp "$ROOT_DIR/config/grid/grid_teacher4x_train.yaml" \
  "$GRID_ROOT/configs/experiment/grid_teacher4x_train.yaml"
cp "$ROOT_DIR/config/grid/rqkmeans_inference_l3.yaml" \
  "$GRID_ROOT/configs/experiment/rqkmeans_inference_l3.yaml"

echo "GRID ready at $GRID_ROOT ($GRID_COMMIT)"
