#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ./scripts/svdgpt_env.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export RUN_ID="${RUN_ID:-svdgpt_gpu_$(date -u +%Y%m%d_%H%M%S)}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_svdgpt.py
