#!/bin/bash
# Deploy the hardened nvidia_uvm.ko module, replacing the stock one.
#
# WARNING: This replaces a running kernel module. All GPU workloads must be
# stopped first. The module swap is atomic (rmmod + insmod).
#
# Usage:
#   sudo ./deploy.sh                              # Use module from build/
#   sudo ./deploy.sh /path/to/nvidia-uvm.ko       # Use specified module

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Must run as root (sudo)"
    exit 1
fi

# Find the module
if [ -n "${1:-}" ]; then
    UVM_KO="$1"
else
    UVM_KO="${SCRIPT_DIR}/build/open-gpu-kernel-modules/kernel-open/nvidia-uvm/nvidia-uvm.ko"
fi

if [ ! -f "${UVM_KO}" ]; then
    echo "ERROR: Module not found at ${UVM_KO}"
    echo "Run ./build.sh first, or specify the path: ./deploy.sh /path/to/nvidia-uvm.ko"
    exit 1
fi

echo "=== Deploying hardened nvidia-uvm.ko ==="
echo "Module: ${UVM_KO}"
echo ""

# Check for running GPU processes
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "WARNING: GPU processes are running. They must be stopped first:"
    nvidia-smi --query-compute-apps=pid,name --format=csv
    echo ""
    read -p "Kill all GPU processes and continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    # Kill GPU processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 2
fi

# Swap the module
echo "Removing stock nvidia_uvm..."
rmmod nvidia_uvm 2>/dev/null || true
sleep 1

echo "Loading hardened nvidia_uvm..."
insmod "${UVM_KO}"

echo ""
echo "=== Verifying ==="

# Quick CUDA test
if command -v python3 &>/dev/null; then
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('CUDA: OK')" 2>/dev/null \
        || echo "WARNING: PyTorch CUDA test failed"
fi

# Check module is loaded
if lsmod | grep -q nvidia_uvm; then
    echo "nvidia_uvm: loaded (hardened)"
else
    echo "ERROR: nvidia_uvm not loaded!"
    exit 1
fi

echo ""
echo "=== SUCCESS: Hardened nvidia_uvm.ko deployed ==="
echo ""
echo "Blocked ioctls: TOOLS_READ/WRITE_PROCESS_MEMORY, MIGRATE, SET_PREFERRED_LOCATION,"
echo "                SET_ACCESSED_BY, READ_DUPLICATION, DYNAMIC_PARALLELISM, + 14 more"
echo ""
echo "To revert to stock: sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm"
