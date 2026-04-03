#!/bin/bash
set -euo pipefail

# ========= BrainVista Inference (arXiv: 2602.04512) =========
# Long-horizon autoregressive rollout prediction

# ========= Paths (modify these) =========
DATA_ROOT="/path/to/data"

VIDEO_ROOT="$DATA_ROOT/features/video_2hz"
AUDIO_ROOT="$DATA_ROOT/features/audio_2hz"
TEXT_ROOT="$DATA_ROOT/features/text_2hz"
Z_ROOT="./z_cache"

TEST_LIST="$DATA_ROOT/test_list.txt"

TOKENIZER_CKPT_DIR="."
TOKENIZER_NAME_TEMPLATE="ckpt_tokenizer_{parcel}/best.pt"

PRED_CKPT="./ckpt_predictor/predictor_best.pt"
OUT_DIR="./predictions"

echo "=============================================="
echo "BrainVista Inference"
echo "  PRED_CKPT:  $PRED_CKPT"
echo "  TEST_LIST:  $TEST_LIST"
echo "  OUT_DIR:    $OUT_DIR"
echo "=============================================="

python -u inference.py \
  --predictor_ckpt "$PRED_CKPT" \
  --video_root "$VIDEO_ROOT" \
  --audio_root "$AUDIO_ROOT" \
  --text_root  "$TEXT_ROOT"  \
  --z_root     "$Z_ROOT"     \
  --run_list_file "$TEST_LIST" \
  --subjects sub-01 \
  --parcels Cont DorsAttn SalVentAttn Limbic SomMot Default Vis \
  --tokenizer_ckpt_dir "$TOKENIZER_CKPT_DIR" \
  --tokenizer_name_template "$TOKENIZER_NAME_TEMPLATE" \
  --K_horizon 20 \
  --out_dir "$OUT_DIR"
