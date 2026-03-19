#!/usr/bin/env bash
set -euo pipefail

# Shared architecture/training knobs for both MLX and CUDA runs.
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export NUM_LAYERS="${NUM_LAYERS:-12}"
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-3}"
export TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"

export ITERATIONS="${ITERATIONS:-20000}"
export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"

export SVD_EVERY="${SVD_EVERY:-8}"
export SVD_MIX="${SVD_MIX:-0.25}"
export SVD_RANK_FC="${SVD_RANK_FC:-64}"
export SVD_RANK_PROJ="${SVD_RANK_PROJ:-64}"
export SVD_RANK_ATTN_PROJ="${SVD_RANK_ATTN_PROJ:-64}"
