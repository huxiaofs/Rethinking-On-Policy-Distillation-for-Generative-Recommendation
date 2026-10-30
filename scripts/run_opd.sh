#!/usr/bin/env bash
set -euo pipefail
# Run one of: reverse, forward, reverse_anchor, forward_anchor.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
METHOD="${METHOD:-forward_anchor}"
case "$METHOD" in
  reverse) direction=reverse; anchor=false ;;
  forward) direction=forward; anchor=false ;;
  reverse_anchor) direction=reverse; anchor=true ;;
  forward_anchor) direction=forward; anchor=true ;;
  *) echo "Unknown METHOD: $METHOD" >&2; exit 2 ;;
esac

: "${STUDENT_CHECKPOINT:?Set STUDENT_CHECKPOINT to the 5k student checkpoint}"
: "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the 30k teacher checkpoint}"
SID_PATH="${SID_PATH:-$OUTPUT_ROOT/${DATASET}_sids/pickle/merged_predictions_tensor_int64.pt}"
RUN_DIR="$OUTPUT_ROOT/${DATASET}_opd_${METHOD}"

cd "$ROOT_DIR"
"$PYTHON_BIN" -m rethinking_opd.grid_train \
  experiment=tiger_train_flat data_dir="$DATA_DIR" \
  semantic_id_path="$SID_PATH" num_hierarchies=3 \
  ckpt_path="$STUDENT_CHECKPOINT" \
  +opd.teacher_checkpoint="$TEACHER_CHECKPOINT" \
  +opd.direction="$direction" +opd.use_target_anchor="$anchor" \
  +opd.temperature="${TEMPERATURE:-1.0}" \
  +opd.anchor_state_budget_fraction="${ANCHOR_FRACTION:-1.0}" \
  trainer.devices="$NUM_DEVICES" trainer.max_steps="${MAX_STEPS:-7000}" \
  trainer.strategy="$TRAINER_STRATEGY" \
  trainer.val_check_interval=2000 \
  callbacks.early_stopping=null callbacks.restart_job=null \
  callbacks.model_checkpoint.monitor=null \
  callbacks.model_checkpoint.save_top_k=-1 \
  callbacks.model_checkpoint.every_n_train_steps=1000 \
  data_loading.train_dataloader_config.dataloader.num_workers="$NUM_WORKERS" \
  hydra.run.dir="$RUN_DIR"

echo "OPD run complete: $RUN_DIR"
