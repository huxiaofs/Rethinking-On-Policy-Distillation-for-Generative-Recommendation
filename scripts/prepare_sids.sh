#!/usr/bin/env bash
set -euo pipefail
# Generate item embeddings and three-level RQ-KMeans semantic IDs.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-google/flan-t5-xl}"
EMBED_DIR="$OUTPUT_ROOT/${DATASET}_embeddings"
RQ_DIR="$OUTPUT_ROOT/${DATASET}_rqkmeans"
SID_DIR="$OUTPUT_ROOT/${DATASET}_sids"

cd "$GRID_ROOT"
"$PYTHON_BIN" -m src.inference \
  experiment=sem_embeds_inference_flat \
  data_dir="$DATA_DIR" \
  embedding_model="$EMBEDDING_MODEL" \
  trainer.devices="$NUM_DEVICES" \
  trainer.strategy="$TRAINER_STRATEGY" \
  data_loading.datamodule.predict_dataloader_config.num_workers="$NUM_WORKERS" \
  hydra.run.dir="$EMBED_DIR"

"$PYTHON_BIN" -m src.train \
  experiment=rkmeans_train_flat \
  data_dir="$DATA_DIR" \
  embedding_path="$EMBED_DIR/pickle/merged_predictions_tensor.pt" \
  embedding_dim=2048 num_hierarchies=3 codebook_width=256 \
  trainer.devices="$NUM_DEVICES" trainer.max_steps=90 \
  hydra.run.dir="$RQ_DIR"

RQ_CHECKPOINT="${RQ_CHECKPOINT:-$RQ_DIR/checkpoints/checkpoint_000_000090.ckpt}"
"$PYTHON_BIN" -m src.inference \
  experiment=rqkmeans_inference_l3 \
  data_dir="$DATA_DIR" \
  embedding_path="$EMBED_DIR/pickle/merged_predictions_tensor.pt" \
  embedding_dim=2048 num_hierarchies=3 codebook_width=256 \
  ckpt_path="$RQ_CHECKPOINT" trainer.devices="$NUM_DEVICES" \
  trainer.strategy="$TRAINER_STRATEGY" \
  hydra.run.dir="$SID_DIR"

echo "Semantic IDs: $SID_DIR/pickle/merged_predictions_tensor_int64.pt"
