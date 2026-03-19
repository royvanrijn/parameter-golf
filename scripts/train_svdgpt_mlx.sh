#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ./scripts/svdgpt_env.sh

export RUN_ID="${RUN_ID:-svdgpt_mlx_$(date -u +%Y%m%d_%H%M%S)}"
python3 train_svdgpt_mlx.py
