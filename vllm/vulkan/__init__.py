# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Vulkan Backend — GPU compute without nvidia_uvm.

Uses VK_NV_cuda_kernel_launch to run PTX kernels through Vulkan's memory
management path (nvidia.ko RM directly), completely bypassing nvidia_uvm.

Usage:
    from vllm.vulkan import is_available, VulkanDevice, VulkanMemoryAllocator
    from vllm.vulkan import KernelModule, KernelFunction, KernelLauncher
    from vllm.vulkan import LaunchConfig

    if is_available():
        device = VulkanDevice(0)
        print(device.info())

        alloc = VulkanMemoryAllocator(device)
        buf = alloc.allocate_device(1024 * 1024)  # 1MB GPU memory
        print(f"GPU VA: 0x{buf.device_address:x}")

        # Load PTX and launch kernel
        module = KernelModule(device, ptx_source="...")
        func = KernelFunction(device, module, "my_kernel")
        launcher = KernelLauncher(device)

        config = LaunchConfig()
        config.grid_x = 256
        config.block_x = 256
        params = [struct.pack('<Q', buf.device_address), struct.pack('<i', 1024)]
        launcher.launch_sync(func, config, params)
"""

try:
    from vllm.vulkan._vulkan_backend import (
        DeviceInfo,
        KernelFunction,
        KernelLauncher,
        KernelModule,
        LaunchConfig,
        VulkanBuffer,
        VulkanDevice,
        VulkanMemoryAllocator,
        is_available,
    )
except ImportError:
    # Vulkan backend not compiled — provide stubs
    def is_available():
        return False

    DeviceInfo = None
    VulkanDevice = None
    VulkanBuffer = None
    VulkanMemoryAllocator = None
    KernelModule = None
    KernelFunction = None
    KernelLauncher = None
    LaunchConfig = None

__all__ = [
    # C++ backend (requires compilation)
    "is_available",
    "DeviceInfo",
    "VulkanDevice",
    "VulkanBuffer",
    "VulkanMemoryAllocator",
    "KernelModule",
    "KernelFunction",
    "KernelLauncher",
    "LaunchConfig",
    # Python modules (always available)
    "kernel_registry",
    "platform",
    "gemm",
    "distributed",
    "worker",
    "command_buffer_cache",
]
