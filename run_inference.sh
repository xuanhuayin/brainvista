#!/bin/bash
set -euo pipefail

# =============================================================================
# BrainVista Inference (arXiv: 2602.04512)
# Long-horizon autoregressive rollout prediction
#
# Usage:
#   bash run_inference.sh
#
# Before running, modify the paths below to match your data layout.
# =============================================================================

# ========= Data paths (MODIFY THESE) =========
BASE_PATH="/path/to/your/data"

VIDEO_ROOT="$BASE_PATH/features/video_2hz"
AUDIO_ROOT="$BASE_PATH/features/audio_2hz"
TEXT_ROOT="$BASE_PATH/features/text_2hz"
Z_ROOT="$BASE_PATH/circuit_tokens"

TEST_LIST="$BASE_PATH/test_list.txt"

# ========= Tokenizer checkpoints =========
TOKENIZER_CKPT_DIR="$BASE_PATH/checkpoints"
TOKENIZER_NAME_TEMPLATE="ckpt_tokenizer_{parcel}/best.pt"

# ========= Predictor checkpoint =========
PRED_CKPT="$BASE_PATH/checkpoints/predictor_best.pt"

# ========= Output =========
OUT_DIR="./predictions"

# ========= Yeo-7 functional networks =========
PARCELS="Cont DorsAttn SalVentAttn Limbic SomMot Default Vis"

echo "=============================================="
echo "  BrainVista Inference"
echo "  Predictor:  $PRED_CKPT"
echo "  Test list:  $TEST_LIST"
echo "  Output:     $OUT_DIR"
echo "=============================================="

python -u inference.py \
  --predictor_ckpt "$PRED_CKPT" \
  --video_root "$VIDEO_ROOT" \
  --audio_root "$AUDIO_ROOT" \
  --text_root  "$TEXT_ROOT"  \
  --z_root     "$Z_ROOT"     \
  --run_list_file "$TEST_LIST" \
  --subjects sub-01 \
  --parcels $PARCELS \
  --tokenizer_ckpt_dir "$TOKENIZER_CKPT_DIR" \
  --tokenizer_name_template "$TOKENIZER_NAME_TEMPLATE" \
  --K_horizon 20 \
  --out_dir "$OUT_DIR" \
  --ratio_v 3 \
  --ratio_a 3 \
  --ratio_w 3 \
  --collapse_v flatten \
  --collapse_a flatten \
  --collapse_w flatten \
  --layer_pool mean \
  --layer_axis 1 \
  --select_layer_idx -1
