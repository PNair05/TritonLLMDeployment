#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_repo="${1:-${project_dir}/artifacts/triton/model-repository}"

source "${project_dir}/scripts/runpod_env.sh"

if [[ ! -f "${model_repo}/phase3-manifest.json" ]]; then
    echo "Generated model repository not found: ${model_repo}" >&2
    echo "Run: python phase3/prepare_model_repository.py" >&2
    exit 1
fi

if [[ ! -f /app/scripts/launch_triton_server.py ]]; then
    echo "Triton launch script not found. Use the pinned 25.03 TRT-LLM container." >&2
    exit 1
fi

echo "Starting Triton with model repository: ${model_repo}"
exec python /app/scripts/launch_triton_server.py \
    --world_size 1 \
    --model_repo "${model_repo}"
