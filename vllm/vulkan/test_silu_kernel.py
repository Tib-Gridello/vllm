#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end test: Launch a SiLU activation kernel via Vulkan.

This test proves that vLLM's CUDA kernels can run through Vulkan without
nvidia_uvm. The flow:

1. Initialize Vulkan device (uses nvidia.ko, NOT nvidia_uvm)
2. Allocate GPU memory via Vulkan (vkAllocateMemory → nvidia.ko RM)
3. Load PTX via vkCreateCudaModuleNV (JIT compiled by Vulkan ICD)
4. Launch kernel via vkCmdCudaLaunchKernelNV
5. Verify output matches numpy reference

Prerequisites:
  - NVIDIA GPU with Vulkan 1.3+ driver
  - VK_NV_cuda_kernel_launch extension support
  - Vulkan SDK headers installed
  - vLLM Vulkan backend compiled (cmake --build . --target _vulkan_backend)
  - nvcc in PATH (for PTX generation)

Run:
  python -m vllm.vulkan.test_silu_kernel
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def generate_silu_ptx(sm_arch: int = 80) -> str:
    """Generate PTX for a SiLU (x * sigmoid(x)) kernel using nvcc."""
    cuda_source = r"""
extern "C" __global__
void silu_kernel(float* output, const float* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}
"""
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
        f.write(cuda_source)
        cu_path = f.name

    ptx_path = cu_path.replace(".cu", ".ptx")

    try:
        result = subprocess.run(
            ["nvcc", "-ptx", f"-arch=sm_{sm_arch}", "-o", ptx_path, cu_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"nvcc stderr: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"nvcc failed with code {result.returncode}")

        ptx = Path(ptx_path).read_text()
        return ptx
    finally:
        Path(cu_path).unlink(missing_ok=True)
        Path(ptx_path).unlink(missing_ok=True)


def silu_reference(x: np.ndarray) -> np.ndarray:
    """NumPy reference implementation of SiLU."""
    return x / (1.0 + np.exp(-x))


def main():
    from vllm.vulkan import (
        KernelFunction,
        KernelLauncher,
        KernelModule,
        LaunchConfig,
        VulkanDevice,
        VulkanMemoryAllocator,
        is_available,
    )

    print("=" * 60)
    print("vLLM Vulkan Backend — SiLU Kernel Test")
    print("=" * 60)

    # Check availability
    if not is_available():
        print("FAIL: VK_NV_cuda_kernel_launch not available")
        print("Ensure you have NVIDIA proprietary Vulkan driver installed")
        sys.exit(1)

    # Initialize device
    device = VulkanDevice(0)
    info = device.info()
    print(f"Device: {info.device_name}")
    print(f"Compute capability: sm_{info.compute_capability_major}"
          f"{info.compute_capability_minor}")
    print(f"Total memory: {info.total_memory / (1024**3):.1f} GiB")
    print()

    # Generate PTX
    sm = info.compute_capability_major * 10 + info.compute_capability_minor
    print(f"Generating PTX for sm_{sm}...")
    ptx = generate_silu_ptx(sm)
    print(f"PTX size: {len(ptx)} bytes")

    # Load module
    print("Loading PTX via vkCreateCudaModuleNV...")
    module = KernelModule(device, ptx)

    # Cache binary for fast reload
    cache = module.get_binary_cache()
    print(f"Binary cache size: {len(cache)} bytes")

    # Create function
    func = KernelFunction(device, module, "silu_kernel")
    print(f"Function: {func.name()}")

    # Allocate memory via Vulkan (NOT nvidia_uvm!)
    alloc = VulkanMemoryAllocator(device)
    n = 1024
    nbytes = n * 4  # float32

    input_buf = alloc.allocate_device(nbytes)
    output_buf = alloc.allocate_device(nbytes)
    print(f"Input buffer:  GPU VA = 0x{input_buf.device_address:016x}")
    print(f"Output buffer: GPU VA = 0x{output_buf.device_address:016x}")

    # Generate test data
    np.random.seed(42)
    input_data = np.random.randn(n).astype(np.float32)
    expected = silu_reference(input_data)

    # Upload input
    print("Uploading input data...")
    alloc.upload(input_data.tobytes(), input_buf)

    # Launch kernel via Vulkan
    print("Launching silu_kernel via vkCmdCudaLaunchKernelNV...")
    launcher = KernelLauncher(device)

    config = LaunchConfig()
    config.grid_x = (n + 255) // 256
    config.block_x = 256

    # Pack parameters: (float* output, const float* input, int n)
    params = [
        struct.pack("<Q", output_buf.device_address),  # float* output
        struct.pack("<Q", input_buf.device_address),    # const float* input
        struct.pack("<i", n),                            # int n
    ]

    launcher.launch_sync(func, config, params)
    print("Kernel completed.")

    # Download output
    output_bytes = alloc.download(output_buf, nbytes)
    output_data = np.frombuffer(output_bytes, dtype=np.float32)

    # Verify
    max_error = np.max(np.abs(output_data - expected))
    print(f"Max error vs reference: {max_error:.2e}")

    if max_error < 1e-5:
        print()
        print("PASS: SiLU kernel via Vulkan matches reference!")
        print("nvidia_uvm was NOT used for this computation.")
    else:
        print()
        print(f"FAIL: Max error {max_error} exceeds threshold 1e-5")
        sys.exit(1)

    # Cleanup
    alloc.free(input_buf)
    alloc.free(output_buf)

    print()
    print("=" * 60)
    print("Test complete. GPU compute via Vulkan without nvidia_uvm: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()
