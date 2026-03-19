#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/openai/parameter-golf.git}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/parameter-golf}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
python3 data/cached_challenge_fineweb.py --variant "${DATA_VARIANT:-sp1024}" --train-shards "${TRAIN_SHARDS:-10}"

echo "RunPod bootstrap complete at ${REPO_DIR}"
