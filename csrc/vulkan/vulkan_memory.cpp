// SPDX-License-Identifier: Apache-2.0
// vLLM Vulkan Backend - GPU Memory Allocation

#include "vulkan_memory.h"
#include "vulkan_device.h"

#include <cstring>
#include <stdexcept>

namespace vllm {
namespace vulkan {

#define VK_CHECK(result, msg)                                     \
  do {                                                            \
    VkResult err = (result);                                      \
    if (err != VK_SUCCESS) {                                      \
      throw std::runtime_error(std::string(msg) +                 \
                               ": VkResult=" + std::to_string(err)); \
    }                                                             \
  } while (0)

VulkanMemoryAllocator::VulkanMemoryAllocator(VulkanDevice& device)
    : device_(device) {
  // Pre-find memory type indices
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(device_.physical_device(), &mem_props);

  // Device-local memory (GPU VRAM)
  device_local_memory_type_ = find_memory_type(
      UINT32_MAX, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Host-visible + host-coherent memory (for staging/uploads)
  host_visible_memory_type_ = find_memory_type(
      UINT32_MAX,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Create a reusable transfer command buffer
  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = device_.command_pool();
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(device_.device(), &alloc_info,
                                     &transfer_cmd_),
           "Failed to allocate transfer command buffer");

  // Create transfer fence
  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VK_CHECK(vkCreateFence(device_.device(), &fence_info, nullptr,
                          &transfer_fence_),
           "Failed to create transfer fence");
}

VulkanMemoryAllocator::~VulkanMemoryAllocator() {
  if (transfer_fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_.device(), transfer_fence_, nullptr);
  }
  // transfer_cmd_ is freed when the command pool is destroyed
}

uint32_t VulkanMemoryAllocator::find_memory_type(
    uint32_t type_filter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(device_.physical_device(), &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type");
}

VulkanBuffer VulkanMemoryAllocator::allocate_device(VkDeviceSize size) {
  VulkanBuffer buf;
  buf.size = size;

  // Create buffer with SHADER_DEVICE_ADDRESS_BIT so we can get a GPU VA
  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VK_CHECK(vkCreateBuffer(device_.device(), &buffer_info, nullptr,
                           &buf.buffer),
           "Failed to create device buffer");

  // Get memory requirements
  VkMemoryRequirements mem_req;
  vkGetBufferMemoryRequirements(device_.device(), buf.buffer, &mem_req);

  // Allocate with DEVICE_ADDRESS_BIT
  VkMemoryAllocateFlagsInfo flags_info = {};
  flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &flags_info;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex =
      find_memory_type(mem_req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VK_CHECK(vkAllocateMemory(device_.device(), &alloc_info, nullptr,
                              &buf.memory),
           "Failed to allocate device memory");

  VK_CHECK(vkBindBufferMemory(device_.device(), buf.buffer, buf.memory, 0),
           "Failed to bind buffer memory");

  // Get the GPU virtual address — this is the key!
  // This address is in the same address space as CUDA pointers on NVIDIA.
  VkBufferDeviceAddressInfo addr_info = {};
  addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  addr_info.buffer = buf.buffer;
  buf.device_address = vkGetBufferDeviceAddress(device_.device(), &addr_info);

  total_allocated_ += size;
  return buf;
}

VulkanBuffer VulkanMemoryAllocator::allocate_staging(VkDeviceSize size) {
  VulkanBuffer buf;
  buf.size = size;

  VkBufferCreateInfo buffer_info = {};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VK_CHECK(vkCreateBuffer(device_.device(), &buffer_info, nullptr,
                           &buf.buffer),
           "Failed to create staging buffer");

  VkMemoryRequirements mem_req;
  vkGetBufferMemoryRequirements(device_.device(), buf.buffer, &mem_req);

  VkMemoryAllocateFlagsInfo flags_info = {};
  flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &flags_info;
  alloc_info.allocationSize = mem_req.size;
  alloc_info.memoryTypeIndex = find_memory_type(
      mem_req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VK_CHECK(vkAllocateMemory(device_.device(), &alloc_info, nullptr,
                              &buf.memory),
           "Failed to allocate staging memory");

  VK_CHECK(vkBindBufferMemory(device_.device(), buf.buffer, buf.memory, 0),
           "Failed to bind staging buffer memory");

  // Map for CPU access
  VK_CHECK(vkMapMemory(device_.device(), buf.memory, 0, size, 0,
                        &buf.mapped_ptr),
           "Failed to map staging memory");

  // Get device address
  VkBufferDeviceAddressInfo addr_info = {};
  addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  addr_info.buffer = buf.buffer;
  buf.device_address = vkGetBufferDeviceAddress(device_.device(), &addr_info);

  return buf;
}

void VulkanMemoryAllocator::free(VulkanBuffer& buf) {
  if (buf.mapped_ptr) {
    vkUnmapMemory(device_.device(), buf.memory);
    buf.mapped_ptr = nullptr;
  }
  if (buf.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_.device(), buf.buffer, nullptr);
    buf.buffer = VK_NULL_HANDLE;
  }
  if (buf.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device_.device(), buf.memory, nullptr);
    buf.memory = VK_NULL_HANDLE;
  }
  total_allocated_ -= buf.size;
  buf.device_address = 0;
  buf.size = 0;
}

void VulkanMemoryAllocator::execute_transfer(VkCommandBuffer cmd) {
  VK_CHECK(vkEndCommandBuffer(cmd), "Failed to end transfer command buffer");

  VkSubmitInfo submit = {};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;

  VK_CHECK(vkResetFences(device_.device(), 1, &transfer_fence_),
           "Failed to reset transfer fence");
  VK_CHECK(vkQueueSubmit(device_.compute_queue(), 1, &submit, transfer_fence_),
           "Failed to submit transfer");
  VK_CHECK(vkWaitForFences(device_.device(), 1, &transfer_fence_, VK_TRUE,
                            UINT64_MAX),
           "Failed waiting for transfer");
}

void VulkanMemoryAllocator::upload(const void* src, VulkanBuffer& dst,
                                    VkDeviceSize size) {
  // Allocate a temporary staging buffer
  VulkanBuffer staging = allocate_staging(size);
  memcpy(staging.mapped_ptr, src, size);

  // Reset command buffer before reuse
  VK_CHECK(vkResetCommandBuffer(transfer_cmd_, 0),
           "Failed to reset transfer command buffer");

  // Record copy command
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(transfer_cmd_, &begin_info),
           "Failed to begin transfer command buffer");

  VkBufferCopy copy_region = {};
  copy_region.size = size;
  vkCmdCopyBuffer(transfer_cmd_, staging.buffer, dst.buffer, 1, &copy_region);

  execute_transfer(transfer_cmd_);
  free(staging);
}

void VulkanMemoryAllocator::download(const VulkanBuffer& src, void* dst,
                                      VkDeviceSize size) {
  VulkanBuffer staging = allocate_staging(size);

  VK_CHECK(vkResetCommandBuffer(transfer_cmd_, 0),
           "Failed to reset transfer command buffer");

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(transfer_cmd_, &begin_info),
           "Failed to begin transfer command buffer");

  VkBufferCopy copy_region = {};
  copy_region.size = size;
  vkCmdCopyBuffer(transfer_cmd_, src.buffer, staging.buffer, 1, &copy_region);

  execute_transfer(transfer_cmd_);
  memcpy(dst, staging.mapped_ptr, size);
  free(staging);
}

void VulkanMemoryAllocator::copy(const VulkanBuffer& src, VulkanBuffer& dst,
                                  VkDeviceSize size) {
  VK_CHECK(vkResetCommandBuffer(transfer_cmd_, 0),
           "Failed to reset transfer command buffer");

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(transfer_cmd_, &begin_info),
           "Failed to begin transfer command buffer");

  VkBufferCopy copy_region = {};
  copy_region.size = size;
  vkCmdCopyBuffer(transfer_cmd_, src.buffer, dst.buffer, 1, &copy_region);

  execute_transfer(transfer_cmd_);
}

}  // namespace vulkan
}  // namespace vllm
