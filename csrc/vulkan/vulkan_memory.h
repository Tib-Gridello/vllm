// SPDX-License-Identifier: Apache-2.0
// vLLM Vulkan Backend - GPU Memory Allocation without nvidia_uvm
//
// Allocates GPU memory through Vulkan's path (nvidia.ko RM directly),
// completely bypassing nvidia_uvm. Returns GPU virtual addresses via
// vkGetBufferDeviceAddress() that are compatible with PTX kernels.

#pragma once

#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vllm {
namespace vulkan {

class VulkanDevice;

// A GPU memory allocation via Vulkan. The device_address() is a GPU virtual
// address that can be passed directly to PTX kernels launched via
// VK_NV_cuda_kernel_launch. On NVIDIA hardware, this is the same address
// space as CUDA pointers.
struct VulkanBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceAddress device_address = 0;  // GPU virtual address for kernel params
  VkDeviceSize size = 0;
  void* mapped_ptr = nullptr;  // Non-null for host-visible memory
};

// Memory allocator that uses Vulkan (nvidia.ko) instead of CUDA (nvidia_uvm).
class VulkanMemoryAllocator {
 public:
  explicit VulkanMemoryAllocator(VulkanDevice& device);
  ~VulkanMemoryAllocator();

  // Allocate device-local GPU memory. Returns a buffer with a valid
  // device_address that can be used in kernel parameters.
  VulkanBuffer allocate_device(VkDeviceSize size);

  // Allocate host-visible, device-accessible staging memory for CPU<->GPU
  // transfers. The returned buffer has a valid mapped_ptr for CPU access
  // and device_address for GPU access.
  VulkanBuffer allocate_staging(VkDeviceSize size);

  // Free a previously allocated buffer.
  void free(VulkanBuffer& buffer);

  // Copy data from host memory to a device buffer using a staging buffer.
  void upload(const void* src, VulkanBuffer& dst, VkDeviceSize size);

  // Copy data from device buffer to host memory using a staging buffer.
  void download(const VulkanBuffer& src, void* dst, VkDeviceSize size);

  // Copy between two device buffers.
  void copy(const VulkanBuffer& src, VulkanBuffer& dst, VkDeviceSize size);

  // Get total allocated memory in bytes
  VkDeviceSize total_allocated() const { return total_allocated_; }

 private:
  uint32_t find_memory_type(uint32_t type_filter,
                            VkMemoryPropertyFlags properties);
  void execute_transfer(VkCommandBuffer cmd);

  VulkanDevice& device_;
  VkCommandBuffer transfer_cmd_ = VK_NULL_HANDLE;
  VkFence transfer_fence_ = VK_NULL_HANDLE;
  VkDeviceSize total_allocated_ = 0;

  uint32_t device_local_memory_type_ = UINT32_MAX;
  uint32_t host_visible_memory_type_ = UINT32_MAX;
};

}  // namespace vulkan
}  // namespace vllm
