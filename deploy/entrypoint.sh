#!/bin/bash
# Entrypoint: start SSH, then either serve vLLM or sleep for manual testing
set -e

# SSH setup
if [ -n "${SSH_PUB_KEY:-}" ]; then
    echo "$SSH_PUB_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd 2>/dev/null || true

echo "=== vLLM Hardened Container ==="
echo "SSH: enabled"

# Mode selection
MODE="${MODE:-serve}"

if [ "$MODE" = "serve" ]; then
    MODEL="${MODEL:-facebook/opt-125m}"
    echo "Mode: serve (model: $MODEL)"
    exec vllm serve "$MODEL" \
        --max-model-len "${MAX_MODEL_LEN:-2048}" \
        --dtype "${DTYPE:-auto}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
        --host 0.0.0.0 --port 8000 \
        --disable-log-requests \
        ${EXTRA_ARGS:-}
elif [ "$MODE" = "test" ]; then
    echo "Mode: test (running E2E test suite)"
    exec /workspace/test_hardened.sh
else
    echo "Mode: shell (sleeping, SSH in to work)"
    exec sleep infinity
fi
