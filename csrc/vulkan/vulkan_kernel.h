// SPDX-License-Identifier: Apache-2.0
// vLLM Vulkan Backend - PTX Kernel Loading & Launch via VK_NV_cuda_kernel_launch
//
// Loads PTX code via vkCreateCudaModuleNV, creates kernel functions via
// vkCreateCudaFunctionNV, and launches them via vkCmdCudaLaunchKernelNV.
// Kernels execute on Vulkan-allocated memory (no nvidia_uvm).

#pragma once

#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vllm {
namespace vulkan {

class VulkanDevice;

// A loaded PTX module with cached binary for fast reload.
class KernelModule {
 public:
  // Load PTX source code as a CUDA module via Vulkan.
  // The Vulkan ICD JIT-compiles the PTX on first load.
  KernelModule(VulkanDevice& device, const std::string& ptx_source);

  // Load from a previously cached binary (skips JIT compilation).
  KernelModule(VulkanDevice& device, const void* cache_data,
               size_t cache_size);

  ~KernelModule();

  KernelModule(const KernelModule&) = delete;
  KernelModule& operator=(const KernelModule&) = delete;
  KernelModule(KernelModule&& other) noexcept;
  KernelModule& operator=(KernelModule&& other) noexcept;

  // Get the compiled binary cache for saving to disk.
  // Feed this back into the cache constructor for fast reload.
  std::vector<uint8_t> get_binary_cache() const;

  VkCudaModuleNV module() const { return module_; }

 private:
  VulkanDevice* device_;
  VkCudaModuleNV module_ = VK_NULL_HANDLE;
};

// A specific kernel function within a module, ready to launch.
class KernelFunction {
 public:
  // Create a function handle for the named entry point in the module.
  KernelFunction(VulkanDevice& device, KernelModule& module,
                 const std::string& function_name);
  ~KernelFunction();

  KernelFunction(const KernelFunction&) = delete;
  KernelFunction& operator=(const KernelFunction&) = delete;
  KernelFunction(KernelFunction&& other) noexcept;
  KernelFunction& operator=(KernelFunction&& other) noexcept;

  VkCudaFunctionNV function() const { return function_; }
  const std::string& name() const { return name_; }

 private:
  VulkanDevice* device_;
  VkCudaFunctionNV function_ = VK_NULL_HANDLE;
  std::string name_;
};

// Launch configuration for a kernel dispatch.
struct LaunchConfig {
  uint32_t grid_x = 1;
  uint32_t grid_y = 1;
  uint32_t grid_z = 1;
  uint32_t block_x = 1;
  uint32_t block_y = 1;
  uint32_t block_z = 1;
  uint32_t shared_mem_bytes = 0;
};

// Records and submits kernel launches via VK_NV_cuda_kernel_launch.
// Manages command buffer recording and synchronization.
class KernelLauncher {
 public:
  explicit KernelLauncher(VulkanDevice& device);
  ~KernelLauncher();

  KernelLauncher(const KernelLauncher&) = delete;
  KernelLauncher& operator=(const KernelLauncher&) = delete;

  // Launch a kernel synchronously. Blocks until completion.
  // params: array of pointers to parameter values (same as cuLaunchKernel).
  //   For a kernel taking (float* a, float* b, int n):
  //     VkDeviceAddress a_addr = ...;
  //     VkDeviceAddress b_addr = ...;
  //     int n = 1024;
  //     const void* params[] = {&a_addr, &b_addr, &n};
  //     launcher.launch_sync(func, config, params, 3);
  void launch_sync(const KernelFunction& func, const LaunchConfig& config,
                   const void* const* params, size_t param_count);

  // Begin recording kernel launches into a command buffer.
  // Call record_launch() for each kernel, then submit().
  void begin_recording();

  // Record a kernel launch into the current command buffer.
  void record_launch(const KernelFunction& func, const LaunchConfig& config,
                     const void* const* params, size_t param_count);

  // Submit all recorded launches and wait for completion.
  void submit_and_wait();

  // Submit all recorded launches without waiting (async).
  // Call wait() later to synchronize.
  void submit();

  // Wait for the last submitted batch to complete.
  void wait();

 private:
  VulkanDevice& device_;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  bool recording_ = false;
};

}  // namespace vulkan
}  // namespace vllm
