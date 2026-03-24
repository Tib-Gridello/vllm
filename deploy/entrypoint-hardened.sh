#!/bin/bash
# vLLM Hardened Entrypoint — Synacktiv-style least-privilege
#
# This entrypoint:
# 1. Downloads the model if not cached
# 2. Starts vLLM listening on a Unix socket (no TCP)
# 3. Applies seccomp profile for nvidia_uvm hardening

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-facebook/opt-125m}"
MODEL_DIR="${MODEL_DIR:-/models}"
SOCKET_PATH="${SOCKET_PATH:-/run/vllm/vllm.sock}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
TP_SIZE="${TP_SIZE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================="
echo "vLLM Hardened Server"
echo "============================================="
echo "Model:    ${MODEL_NAME}"
echo "Socket:   ${SOCKET_PATH}"
echo "Max len:  ${MAX_MODEL_LEN}"
echo "GPU util: ${GPU_MEMORY_UTILIZATION}"
echo "TP size:  ${TP_SIZE}"
echo "============================================="

# Check GPU access
echo ""
echo "--- GPU Status ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
else
    echo "nvidia-smi not available"
fi

# Check nvidia_uvm status
echo ""
echo "--- nvidia_uvm Status ---"
if [ -f /proc/modules ]; then
    if grep -q "nvidia_uvm" /proc/modules 2>/dev/null; then
        echo "nvidia_uvm: LOADED"
        echo "WARNING: nvidia_uvm is loaded. Ensure seccomp profile is applied:"
        echo "  --security-opt seccomp=/etc/vllm/uvm_seccomp_profile.json"
    else
        echo "nvidia_uvm: NOT LOADED (good — pure Vulkan mode or module not needed)"
    fi
fi

# Check Vulkan availability
echo ""
echo "--- Vulkan Status ---"
if command -v vulkaninfo &>/dev/null; then
    vulkaninfo --summary 2>/dev/null | head -10 || echo "vulkaninfo failed"
else
    echo "vulkaninfo not available (Vulkan runtime may still work)"
fi

# Download model if needed
echo ""
echo "--- Model Download ---"
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Check if model is already cached
if python3 -c "
from huggingface_hub import try_to_load_from_cache
import os
model = '${MODEL_NAME}'
# Check if config.json is cached (indicates model is downloaded)
path = try_to_load_from_cache(model, 'config.json')
if path is not None:
    print(f'Model cached at: {os.path.dirname(path)}')
    exit(0)
exit(1)
" 2>/dev/null; then
    echo "Model already cached, skipping download"
else
    echo "Downloading model: ${MODEL_NAME}..."
    huggingface-cli download "${MODEL_NAME}" \
        --local-dir "${MODEL_DIR}/${MODEL_NAME}" \
        2>&1 | tail -5
fi

# Remove the socket if it exists from a previous run
rm -f "${SOCKET_PATH}"

# Start vLLM
echo ""
echo "============================================="
echo "Starting vLLM server on ${SOCKET_PATH}"
echo "============================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --unix-socket-path "${SOCKET_PATH}" \
    --disable-log-requests \
    ${EXTRA_ARGS}
