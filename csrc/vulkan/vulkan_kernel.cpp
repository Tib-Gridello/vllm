// SPDX-License-Identifier: Apache-2.0
// vLLM Vulkan Backend - PTX Kernel Loading & Launch

#include "vulkan_kernel.h"
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

// ============================================================================
// KernelModule
// ============================================================================

KernelModule::KernelModule(VulkanDevice& device, const std::string& ptx_source)
    : device_(&device) {
  VkCudaModuleCreateInfoNV create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV;
  create_info.dataSize = ptx_source.size();
  create_info.pData = ptx_source.data();

  VK_CHECK(device_->fp_vkCreateCudaModuleNV_(device_->device(), &create_info,
                                               nullptr, &module_),
           "Failed to create CUDA module from PTX");
}

KernelModule::KernelModule(VulkanDevice& device, const void* cache_data,
                           size_t cache_size)
    : device_(&device) {
  VkCudaModuleCreateInfoNV create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV;
  create_info.dataSize = cache_size;
  create_info.pData = cache_data;

  VK_CHECK(device_->fp_vkCreateCudaModuleNV_(device_->device(), &create_info,
                                               nullptr, &module_),
           "Failed to create CUDA module from cache");
}

KernelModule::~KernelModule() {
  if (module_ != VK_NULL_HANDLE && device_) {
    device_->fp_vkDestroyCudaModuleNV_(device_->device(), module_, nullptr);
  }
}

KernelModule::KernelModule(KernelModule&& other) noexcept
    : device_(other.device_), module_(other.module_) {
  other.module_ = VK_NULL_HANDLE;
}

KernelModule& KernelModule::operator=(KernelModule&& other) noexcept {
  if (this != &other) {
    if (module_ != VK_NULL_HANDLE && device_) {
      device_->fp_vkDestroyCudaModuleNV_(device_->device(), module_, nullptr);
    }
    device_ = other.device_;
    module_ = other.module_;
    other.module_ = VK_NULL_HANDLE;
  }
  return *this;
}

std::vector<uint8_t> KernelModule::get_binary_cache() const {
  // First call: get size
  size_t cache_size = 0;
  VK_CHECK(device_->fp_vkGetCudaModuleCacheNV_(device_->device(), module_,
                                                 &cache_size, nullptr),
           "Failed to get module cache size");

  // Second call: get data
  std::vector<uint8_t> cache(cache_size);
  VK_CHECK(device_->fp_vkGetCudaModuleCacheNV_(device_->device(), module_,
                                                 &cache_size, cache.data()),
           "Failed to get module cache data");

  return cache;
}

// ============================================================================
// KernelFunction
// ============================================================================

KernelFunction::KernelFunction(VulkanDevice& device, KernelModule& module,
                               const std::string& function_name)
    : device_(&device), name_(function_name) {
  VkCudaFunctionCreateInfoNV create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV;
  create_info.module = module.module();
  create_info.pName = function_name.c_str();

  VK_CHECK(device_->fp_vkCreateCudaFunctionNV_(device_->device(), &create_info,
                                                 nullptr, &function_),
           "Failed to create CUDA function '" + function_name + "'");
}

KernelFunction::~KernelFunction() {
  if (function_ != VK_NULL_HANDLE && device_) {
    device_->fp_vkDestroyCudaFunctionNV_(device_->device(), function_, nullptr);
  }
}

KernelFunction::KernelFunction(KernelFunction&& other) noexcept
    : device_(other.device_),
      function_(other.function_),
      name_(std::move(other.name_)) {
  other.function_ = VK_NULL_HANDLE;
}

KernelFunction& KernelFunction::operator=(KernelFunction&& other) noexcept {
  if (this != &other) {
    if (function_ != VK_NULL_HANDLE && device_) {
      device_->fp_vkDestroyCudaFunctionNV_(device_->device(), function_,
                                            nullptr);
    }
    device_ = other.device_;
    function_ = other.function_;
    name_ = std::move(other.name_);
    other.function_ = VK_NULL_HANDLE;
  }
  return *this;
}

// ============================================================================
// KernelLauncher
// ============================================================================

KernelLauncher::KernelLauncher(VulkanDevice& device) : device_(device) {
  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = device_.command_pool();
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  VK_CHECK(vkAllocateCommandBuffers(device_.device(), &alloc_info, &cmd_),
           "Failed to allocate launch command buffer");

  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VK_CHECK(vkCreateFence(device_.device(), &fence_info, nullptr, &fence_),
           "Failed to create launch fence");
}

KernelLauncher::~KernelLauncher() {
  if (fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_.device(), fence_, nullptr);
  }
}

void KernelLauncher::launch_sync(const KernelFunction& func,
                                  const LaunchConfig& config,
                                  const void* const* params,
                                  size_t param_count) {
  begin_recording();
  record_launch(func, config, params, param_count);
  submit_and_wait();
}

void KernelLauncher::begin_recording() {
  if (recording_) {
    throw std::runtime_error("Already recording");
  }

  VK_CHECK(vkResetCommandBuffer(cmd_, 0),
           "Failed to reset command buffer");

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd_, &begin_info),
           "Failed to begin command buffer");

  recording_ = true;
}

void KernelLauncher::record_launch(const KernelFunction& func,
                                    const LaunchConfig& config,
                                    const void* const* params,
                                    size_t param_count) {
  if (!recording_) {
    throw std::runtime_error("Not recording — call begin_recording() first");
  }

  VkCudaLaunchInfoNV launch_info = {};
  launch_info.sType = VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV;
  launch_info.function = func.function();
  launch_info.gridDimX = config.grid_x;
  launch_info.gridDimY = config.grid_y;
  launch_info.gridDimZ = config.grid_z;
  launch_info.blockDimX = config.block_x;
  launch_info.blockDimY = config.block_y;
  launch_info.blockDimZ = config.block_z;
  launch_info.sharedMemBytes = config.shared_mem_bytes;
  launch_info.paramCount = param_count;
  launch_info.pParams = params;
  launch_info.extraCount = 0;
  launch_info.pExtras = nullptr;

  device_.fp_vkCmdCudaLaunchKernelNV_(cmd_, &launch_info);
}

void KernelLauncher::submit_and_wait() {
  submit();
  wait();
}

void KernelLauncher::submit() {
  if (!recording_) {
    throw std::runtime_error("Not recording");
  }

  VK_CHECK(vkEndCommandBuffer(cmd_), "Failed to end command buffer");
  recording_ = false;

  VK_CHECK(vkResetFences(device_.device(), 1, &fence_),
           "Failed to reset fence");

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd_;

  VK_CHECK(vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence_),
           "Failed to submit kernel launch");
}

void KernelLauncher::wait() {
  VK_CHECK(vkWaitForFences(device_.device(), 1, &fence_, VK_TRUE, UINT64_MAX),
           "Failed waiting for kernel completion");
}

}  // namespace vulkan
}  // namespace vllm
