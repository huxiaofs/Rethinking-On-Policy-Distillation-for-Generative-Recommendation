#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
: "${CHECKPOINT:?Set CHECKPOINT to the student checkpoint to evaluate}"
SID_PATH="${SID_PATH:-$OUTPUT_ROOT/${DATASET}_sids/pickle/merged_predictions_tensor_int64.pt}"
RUN_NAME="${RUN_NAME:-evaluation}"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m rethinking_opd.grid_validate \
  experiment=tiger_train_flat data_dir="$DATA_DIR" \
  semantic_id_path="$SID_PATH" num_hierarchies=3 ckpt_path="$CHECKPOINT" \
  trainer.devices=1 trainer.strategy=auto trainer.num_sanity_val_steps=0 \
  train=false test=false callbacks.early_stopping=null \
  callbacks.restart_job=null callbacks.model_checkpoint=null \
  callbacks.model_summary=null hydra.run.dir="$OUTPUT_ROOT/${DATASET}_${RUN_NAME}"
