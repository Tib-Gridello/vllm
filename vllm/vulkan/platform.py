# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Vulkan Platform — GPU management without nvidia_uvm.

This platform uses Vulkan for device enumeration, memory management, and
kernel execution. It replaces the CUDA platform for deployments that need
to eliminate the nvidia_uvm kernel module.

Device queries go through Vulkan API (which uses nvidia.ko directly),
NOT through CUDA (which loads nvidia_uvm).
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    pass


class VulkanPlatformInfo:
    """Vulkan device information without initializing CUDA."""

    def __init__(self):
        from vllm.vulkan import VulkanDevice, is_available

        if not is_available():
            raise RuntimeError(
                "Vulkan backend not available. Ensure NVIDIA proprietary "
                "Vulkan driver is installed with VK_NV_cuda_kernel_launch.")

        self._devices = VulkanDevice.enumerate_devices()

    @cache
    def device_count(self) -> int:
        return len(self._devices)

    @cache
    def get_device_name(self, device_id: int = 0) -> str:
        return self._devices[device_id].device_name

    @cache
    def get_compute_capability(self, device_id: int = 0) -> tuple[int, int]:
        d = self._devices[device_id]
        return (d.compute_capability_major, d.compute_capability_minor)

    @cache
    def get_total_memory(self, device_id: int = 0) -> int:
        return self._devices[device_id].total_memory

    def is_cuda_kernel_launch_supported(self, device_id: int = 0) -> bool:
        return self._devices[device_id].cuda_kernel_launch_supported

    def print_device_info(self):
        """Print all Vulkan device info (for diagnostics)."""
        for d in self._devices:
            logger.info(
                "Vulkan device %d: %s (sm_%d%d, %.1f GiB, cuda_launch=%s)",
                d.device_id, d.device_name,
                d.compute_capability_major, d.compute_capability_minor,
                d.total_memory / (1024**3),
                d.cuda_kernel_launch_supported)


class VulkanDeviceContext:
    """
    Manages a Vulkan device, memory allocator, and kernel manager.

    This is the main entry point for the Vulkan backend. One context per GPU.
    """

    def __init__(self, device_id: int = 0):
        from vllm.vulkan import (
            KernelLauncher,
            VulkanDevice,
            VulkanMemoryAllocator,
        )
        from vllm.vulkan.kernel_registry import VulkanKernelManager

        self.device_id = device_id
        self.device = VulkanDevice(device_id)
        self.allocator = VulkanMemoryAllocator(self.device)
        self.launcher = KernelLauncher(self.device)

        info = self.device.info()
        sm = info.compute_capability_major * 10 + info.compute_capability_minor
        self.kernel_manager = VulkanKernelManager(self.device, sm)

        logger.info(
            "VulkanDeviceContext initialized: %s (sm_%d, %.1f GiB)",
            info.device_name, sm, info.total_memory / (1024**3))

    def launch_kernel(self, op_name: str, dtype: str,
                      grid: tuple[int, int, int],
                      block: tuple[int, int, int],
                      *args, shared_mem: int = 0):
        """
        Launch a registered kernel.

        Args:
            op_name: Operation name (e.g. "silu_and_mul")
            dtype: Data type (e.g. "f16", "bf16", "f32")
            grid: (grid_x, grid_y, grid_z)
            block: (block_x, block_y, block_z)
            *args: Kernel parameters (pointers as int, scalars as int/float)
            shared_mem: Shared memory bytes
        """
        from vllm.vulkan import LaunchConfig

        func, spec = self.kernel_manager.get_function(op_name, dtype)
        params = spec.pack_params(*args)

        config = LaunchConfig()
        config.grid_x, config.grid_y, config.grid_z = grid
        config.block_x, config.block_y, config.block_z = block
        config.shared_mem_bytes = shared_mem

        self.launcher.launch_sync(func, config, params)

    def allocate(self, size: int):
        """Allocate device-local GPU memory. Returns VulkanBuffer."""
        return self.allocator.allocate_device(size)

    def free(self, buffer):
        """Free a VulkanBuffer."""
        self.allocator.free(buffer)

    def upload(self, data: bytes, dst_buffer):
        """Upload bytes from CPU to GPU buffer."""
        self.allocator.upload(data, dst_buffer)

    def download(self, src_buffer, size: int) -> bytes:
        """Download bytes from GPU buffer to CPU."""
        return self.allocator.download(src_buffer, size)
