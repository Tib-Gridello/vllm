// SPDX-License-Identifier: Apache-2.0
// vLLM Vulkan Backend - Device initialization without nvidia_uvm
//
// This module initializes Vulkan with VK_NV_cuda_kernel_launch support,
// enabling PTX kernel execution through Vulkan's memory management path
// which goes through nvidia.ko directly, bypassing nvidia_uvm entirely.

#pragma once

#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vllm {
namespace vulkan {

struct DeviceInfo {
  uint32_t device_id;
  std::string device_name;
  uint32_t compute_capability_major;
  uint32_t compute_capability_minor;
  uint64_t total_memory;
  uint32_t max_compute_work_group_count[3];
  uint32_t max_compute_work_group_size[3];
  uint32_t max_compute_shared_memory_size;
  bool cuda_kernel_launch_supported;
  bool buffer_device_address_supported;
};

// Manages a single Vulkan device with VK_NV_cuda_kernel_launch support.
// One VulkanDevice per physical GPU.
class VulkanDevice {
 public:
  // Initialize Vulkan device for the given physical device index.
  // Enables VK_NV_cuda_kernel_launch + VK_KHR_buffer_device_address.
  explicit VulkanDevice(uint32_t device_index = 0);
  ~VulkanDevice();

  // Non-copyable, movable
  VulkanDevice(const VulkanDevice&) = delete;
  VulkanDevice& operator=(const VulkanDevice&) = delete;
  VulkanDevice(VulkanDevice&& other) noexcept;
  VulkanDevice& operator=(VulkanDevice&& other) noexcept;

  VkDevice device() const { return device_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkQueue compute_queue() const { return compute_queue_; }
  uint32_t compute_queue_family() const { return compute_queue_family_; }
  VkCommandPool command_pool() const { return command_pool_; }
  const DeviceInfo& info() const { return info_; }

  // Check if the VK_NV_cuda_kernel_launch extension is available
  static bool is_cuda_kernel_launch_supported(uint32_t device_index = 0);

  // Enumerate all available Vulkan physical devices
  static std::vector<DeviceInfo> enumerate_devices();

  // Get the singleton VkInstance (created on first call)
  static VkInstance instance();

 private:
  void init_instance();
  void select_physical_device(uint32_t device_index);
  void create_device();
  void create_command_pool();
  void query_device_info();
  void cleanup();

  static VkInstance instance_;
  static bool instance_created_;

  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue compute_queue_ = VK_NULL_HANDLE;
  uint32_t compute_queue_family_ = 0;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  DeviceInfo info_ = {};

  // Function pointers for VK_NV_cuda_kernel_launch (loaded dynamically)
  PFN_vkCreateCudaModuleNV fp_vkCreateCudaModuleNV_ = nullptr;
  PFN_vkGetCudaModuleCacheNV fp_vkGetCudaModuleCacheNV_ = nullptr;
  PFN_vkCreateCudaFunctionNV fp_vkCreateCudaFunctionNV_ = nullptr;
  PFN_vkDestroyCudaModuleNV fp_vkDestroyCudaModuleNV_ = nullptr;
  PFN_vkDestroyCudaFunctionNV fp_vkDestroyCudaFunctionNV_ = nullptr;
  PFN_vkCmdCudaLaunchKernelNV fp_vkCmdCudaLaunchKernelNV_ = nullptr;

  // Grant access to function pointers from other modules
  friend class KernelModule;
  friend class KernelLauncher;
};

}  // namespace vulkan
}  // namespace vllm
