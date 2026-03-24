#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Extract PTX from vLLM's CUDA kernels for use with VK_NV_cuda_kernel_launch.
#
# This script compiles standalone kernel files to PTX. These kernels are
# extracted from the original .cu files but stripped of PyTorch/ATen
# dependencies, keeping only the raw __global__ kernel functions.
#
# Usage:
#   ./extract_ptx.sh [sm_arch]  # default: sm_80
#   ./extract_ptx.sh 90         # for Hopper

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/generated"
SM_ARCH="${1:-80}"

mkdir -p "${OUTPUT_DIR}"

echo "=== Extracting PTX for sm_${SM_ARCH} ==="

# Compile each standalone kernel to PTX
for cu_file in "${SCRIPT_DIR}"/standalone/*.cu; do
    if [ ! -f "${cu_file}" ]; then
        echo "No standalone kernel files found in ${SCRIPT_DIR}/standalone/"
        exit 1
    fi

    base=$(basename "${cu_file}" .cu)
    ptx_file="${OUTPUT_DIR}/${base}.sm${SM_ARCH}.ptx"

    echo "  Compiling ${base}.cu → ${ptx_file}"
    nvcc -ptx \
        -arch="sm_${SM_ARCH}" \
        -std=c++17 \
        -O3 \
        --use_fast_math \
        -o "${ptx_file}" \
        "${cu_file}" \
        2>&1 | sed 's/^/    /'

    if [ $? -ne 0 ]; then
        echo "  FAILED: ${base}.cu"
    else
        echo "  OK: $(wc -c < "${ptx_file}") bytes"
    fi
done

echo ""
echo "PTX files generated in ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}"/*.ptx 2>/dev/null || echo "No PTX files generated"
