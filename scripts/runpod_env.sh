#!/usr/bin/env bash

# Source this file after activating the project's virtual environment:
#   source scripts/runpod_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script instead of executing it: source scripts/runpod_env.sh" >&2
    exit 1
fi

# Some RunPod CUDA 13 images register a forward-compatibility libcuda before
# the driver library mounted from the host. Consumer RTX GPUs cannot use that
# forward-compatibility path. Prefer the host library when it is available.
if [[ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

# Keep downloaded model weights on RunPod's persistent workspace volume.
if [[ -d /workspace ]]; then
    export HF_HOME="${HF_HOME:-/workspace/huggingface-cache}"
fi

# The base image enables this optional downloader without installing it.
# Hugging Face Hub falls back to its installed download implementation.
unset HF_HUB_ENABLE_HF_TRANSFER

echo "RunPod environment configured"
echo "HF_HOME=${HF_HOME:-<default>}"
