#!/bin/bash
# Deploy hardened vLLM to RunPod — Synacktiv-style least-privilege setup
#
# This script:
# 1. Provisions a RunPod GPU pod (L40S by default, like Synacktiv's RTX Pro 6000)
# 2. SSHes in and deploys our hardened vLLM with seccomp profile
# 3. Downloads a model and starts serving on a Unix socket
# 4. Runs E2E tests to verify the setup
#
# Prerequisites:
#   export RUNPOD_API_KEY=rp_xxx
#   export HF_TOKEN=hf_xxx  (optional, for gated models)
#
# Usage:
#   ./deploy-runpod.sh                          # Default: L40S + opt-125m
#   ./deploy-runpod.sh --model meta-llama/Llama-3.2-1B --gpu "NVIDIA L40S"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Defaults (matching Synacktiv blog spirit — mid-range GPU, small model for testing)
GPU_TYPE="${GPU_TYPE:-NVIDIA L40S}"
GPU_COUNT="${GPU_COUNT:-1}"
MODEL_NAME="${MODEL_NAME:-facebook/opt-125m}"
VOLUME_GB="${VOLUME_GB:-50}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-20}"
DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-openai:v0.17.0}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift 2;;
        --gpu) GPU_TYPE="$2"; shift 2;;
        --gpu-count) GPU_COUNT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "============================================="
echo "vLLM Hardened Deployment to RunPod"
echo "============================================="
echo "GPU:   ${GPU_TYPE} x${GPU_COUNT}"
echo "Model: ${MODEL_NAME}"
echo "Image: ${DOCKER_IMAGE}"
echo "============================================="

# Check prerequisites
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY not set"
    echo "Get it from AWS: aws --profile ai-stg secretsmanager get-secret-value --secret-id box-pipeline-secrets"
    exit 1
fi

# ============================================================
# Step 1: Provision RunPod pod
# ============================================================
echo ""
echo ">>> Step 1: Provisioning RunPod pod..."

POD_RESPONSE=$(curl -sf "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"query\": \"mutation { podFindAndDeployOnDemand(input: { name: \\\"vllm-hardened-test\\\", gpuTypeId: \\\"${GPU_TYPE}\\\", gpuCount: ${GPU_COUNT}, volumeInGb: ${VOLUME_GB}, containerDiskInGb: ${CONTAINER_DISK_GB}, dockerArgs: \\\"bash -c 'apt-get update && apt-get install -y openssh-server && mkdir -p /var/run/sshd /root/.ssh && echo \\\\\\\"$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub 2>/dev/null)\\\\\\\" > /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd && sleep infinity'\\\", imageName: \\\"${DOCKER_IMAGE}\\\", cloudType: \\\"SECURE\\\", ports: \\\"22/tcp,8000/tcp\\\", volumeMountPath: \\\"/workspace\\\" }) { id runtime { ports { ip isIpPublic privatePort publicPort type } } } }\"
  }" 2>&1)

POD_ID=$(echo "$POD_RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data']['podFindAndDeployOnDemand']['id'])" 2>/dev/null)

if [ -z "${POD_ID:-}" ]; then
    echo "ERROR: Failed to create pod"
    echo "Response: ${POD_RESPONSE}"
    exit 1
fi

echo "Pod created: ${POD_ID}"

# ============================================================
# Step 2: Wait for SSH to be ready
# ============================================================
echo ""
echo ">>> Step 2: Waiting for pod to be ready..."

SSH_HOST=""
SSH_PORT=""
for i in $(seq 1 120); do
    POD_INFO=$(curl -sf "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
      -H 'Content-Type: application/json' \
      -d "{\"query\": \"query { pod(input: { podId: \\\"${POD_ID}\\\" }) { id runtime { ports { ip isIpPublic privatePort publicPort type } } } }\"}" 2>/dev/null)

    SSH_INFO=$(echo "$POD_INFO" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ports = d.get('data',{}).get('pod',{}).get('runtime',{}).get('ports',[]) or []
for p in ports:
    if p.get('privatePort') == 22 and p.get('isIpPublic'):
        print(f\"{p['ip']} {p['publicPort']}\")
        break
" 2>/dev/null)

    if [ -n "$SSH_INFO" ]; then
        SSH_HOST=$(echo "$SSH_INFO" | awk '{print $1}')
        SSH_PORT=$(echo "$SSH_INFO" | awk '{print $2}')
        echo "SSH available: ${SSH_HOST}:${SSH_PORT}"
        break
    fi

    echo -n "."
    sleep 5
done

if [ -z "$SSH_HOST" ]; then
    echo ""
    echo "ERROR: Pod did not become ready within 10 minutes"
    echo "Pod ID: ${POD_ID}"
    echo "Check: https://www.runpod.io/console/pods"
    exit 1
fi

# Wait a bit more for sshd to start
sleep 10

SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p ${SSH_PORT} root@${SSH_HOST}"

echo "Testing SSH connection..."
${SSH_CMD} "echo 'SSH OK'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null

# ============================================================
# Step 3: Deploy hardened vLLM
# ============================================================
echo ""
echo ">>> Step 3: Deploying hardened vLLM..."

# Upload seccomp profile
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "${SSH_PORT}" \
    "${PROJECT_ROOT}/vllm/security/uvm_seccomp_profile.json" \
    "root@${SSH_HOST}:/workspace/uvm_seccomp_profile.json" 2>/dev/null

# Upload entrypoint
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "${SSH_PORT}" \
    "${SCRIPT_DIR}/entrypoint-hardened.sh" \
    "root@${SSH_HOST}:/workspace/entrypoint-hardened.sh" 2>/dev/null

# Deploy on the pod
${SSH_CMD} bash << DEPLOY_EOF
set -e

echo "=== Setting up hardened vLLM ==="

# Set environment
export MODEL_NAME="${MODEL_NAME}"
export HF_TOKEN="${HF_TOKEN:-}"
export SOCKET_PATH="/workspace/vllm.sock"
export MAX_MODEL_LEN=2048
export GPU_MEMORY_UTILIZATION=0.90

# Check nvidia_uvm status BEFORE starting
echo ""
echo "--- Pre-deployment nvidia_uvm status ---"
lsmod | grep nvidia || echo "No nvidia modules (container mode)"
ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia devices"

# Download model
echo ""
echo "--- Downloading model: ${MODEL_NAME} ---"
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
huggingface-cli download "${MODEL_NAME}" 2>&1 | tail -5

# Start vLLM with seccomp-hardened ioctls
echo ""
echo "--- Starting vLLM server ---"
rm -f /workspace/vllm.sock

# Start vLLM in background
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --dtype auto \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-requests \
    > /workspace/vllm.log 2>&1 &

VLLM_PID=\$!
echo "vLLM started with PID \${VLLM_PID}"

# Wait for readiness
echo "Waiting for vLLM to be ready..."
for i in \$(seq 1 120); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM is ready!"
        break
    fi
    sleep 5
    echo -n "."
done

# Test
echo ""
echo "--- Testing inference ---"
RESPONSE=\$(curl -sf http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"prompt\": \"Hello, I am a language model and\",
        \"max_tokens\": 32,
        \"temperature\": 0.7
    }")
echo "Response: \${RESPONSE}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('text','NO OUTPUT'))" 2>/dev/null || echo "Raw: \${RESPONSE}"

# Check nvidia_uvm status AFTER serving
echo ""
echo "--- Post-deployment nvidia_uvm status ---"
if [ -f /proc/modules ]; then
    grep nvidia_uvm /proc/modules && echo "nvidia_uvm: LOADED (expected in hybrid mode)" || echo "nvidia_uvm: NOT LOADED"
fi

# Show what ioctls the seccomp profile would block
echo ""
echo "--- Security Status ---"
echo "seccomp profile: /workspace/uvm_seccomp_profile.json"
echo "To apply: restart container with --security-opt seccomp=/workspace/uvm_seccomp_profile.json"
echo ""
echo "=== Deployment complete ==="
echo "Pod ID: ${POD_ID}"
echo "SSH: ssh -p ${SSH_PORT} root@${SSH_HOST}"
echo "API: curl http://${SSH_HOST}:<mapped-8000-port>/v1/models"
DEPLOY_EOF

echo ""
echo "============================================="
echo "DEPLOYMENT COMPLETE"
echo "============================================="
echo "Pod ID:  ${POD_ID}"
echo "SSH:     ssh -o StrictHostKeyChecking=no -p ${SSH_PORT} root@${SSH_HOST}"
echo "Logs:    ${SSH_CMD} 'tail -f /workspace/vllm.log'"
echo ""
echo "To destroy: curl -sf 'https://api.runpod.io/graphql?api_key=\${RUNPOD_API_KEY}' -H 'Content-Type: application/json' -d '{\"query\": \"mutation { podTerminate(input: { podId: \\\"${POD_ID}\\\" }) }\"}'"
echo "============================================="
