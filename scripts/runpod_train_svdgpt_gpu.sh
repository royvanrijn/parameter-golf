#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# For a single H100 pod: NPROC_PER_NODE=1 ./scripts/runpod_train_svdgpt_gpu.sh
# For multi-GPU pods, set NPROC_PER_NODE to your GPU count.
exec ./scripts/train_svdgpt_gpu.sh
