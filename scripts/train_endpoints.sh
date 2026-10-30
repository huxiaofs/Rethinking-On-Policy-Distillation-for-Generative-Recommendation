#!/usr/bin/env bash
set -euo pipefail
# Train the public 13.1M student endpoint and 49M teacher endpoint.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SID_PATH="${SID_PATH:-$OUTPUT_ROOT/${DATASET}_sids/pickle/merged_predictions_tensor_int64.pt}"

train_endpoint() {
  local role="$1" experiment="$2" max_steps="$3"
  local run_dir="$OUTPUT_ROOT/${DATASET}_${role}"
  cd "$GRID_ROOT"
  "$PYTHON_BIN" -m src.train \
    experiment="$experiment" data_dir="$DATA_DIR" \
    semantic_id_path="$SID_PATH" num_hierarchies=3 \
    trainer.devices="$NUM_DEVICES" trainer.max_steps="$max_steps" \
    trainer.strategy="$TRAINER_STRATEGY" \
    trainer.val_check_interval=5000 \
    callbacks.early_stopping=null callbacks.restart_job=null \
    callbacks.model_checkpoint.monitor=null \
    callbacks.model_checkpoint.save_top_k=-1 \
    callbacks.model_checkpoint.every_n_train_steps=5000 \
    data_loading.train_dataloader_config.dataloader.num_workers="$NUM_WORKERS" \
    hydra.run.dir="$run_dir"
}

train_endpoint student tiger_train_flat "${STUDENT_STEPS:-5000}"
train_endpoint teacher grid_teacher4x_train "${TEACHER_STEPS:-30000}"

echo "Student checkpoints: $OUTPUT_ROOT/${DATASET}_student/checkpoints"
echo "Teacher checkpoints: $OUTPUT_ROOT/${DATASET}_teacher/checkpoints"
