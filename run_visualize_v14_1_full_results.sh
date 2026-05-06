#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ubuntu22/decision_on_nuscenes}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/conda_envs/nuscenes/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_DIR/scripts/train_transformer_planning_v14_1_real_camera_cache_enabled_full_pipeline.py}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/outputs_transformer_planning_v14_1_real_camera_cache_enabled/best_score_model.pt}"
METRICS="${METRICS:-$PROJECT_DIR/outputs_transformer_planning_v14_1_real_camera_cache_enabled/metrics.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs_transformer_planning_v14_1_real_camera_cache_enabled/vis_full}"

$PYTHON_BIN /mnt/data/visualize_v14_1_full_results.py \
  --train_script "$TRAIN_SCRIPT" \
  --checkpoint "$CHECKPOINT" \
  --metrics "$METRICS" \
  --output_dir "$OUTPUT_DIR" \
  --traj_samples "${TRAJ_SAMPLES:-48}" \
  --bev_samples "${BEV_SAMPLES:-24}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --save_debug_npz \
  ${EXTRA_ARGS:-}
