// SPDX-License-Identifier: Apache-2.0
// vLLM Vulkan Backend - Device initialization

#include "vulkan_device.h"

#include <algorithm>
#include <cstring>
#include <iostream>
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

VkInstance VulkanDevice::instance_ = VK_NULL_HANDLE;
bool VulkanDevice::instance_created_ = false;

VkInstance VulkanDevice::instance() {
  if (!instance_created_) {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "vllm-vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "vllm";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_),
             "Failed to create Vulkan instance");
    instance_created_ = true;
  }
  return instance_;
}

VulkanDevice::VulkanDevice(uint32_t device_index) {
  init_instance();
  select_physical_device(device_index);
  create_device();
  create_command_pool();
  query_device_info();
}

VulkanDevice::~VulkanDevice() { cleanup(); }

VulkanDevice::VulkanDevice(VulkanDevice&& other) noexcept
    : physical_device_(other.physical_device_),
      device_(other.device_),
      compute_queue_(other.compute_queue_),
      compute_queue_family_(other.compute_queue_family_),
      command_pool_(other.command_pool_),
      info_(std::move(other.info_)),
      fp_vkCreateCudaModuleNV_(other.fp_vkCreateCudaModuleNV_),
      fp_vkGetCudaModuleCacheNV_(other.fp_vkGetCudaModuleCacheNV_),
      fp_vkCreateCudaFunctionNV_(other.fp_vkCreateCudaFunctionNV_),
      fp_vkDestroyCudaModuleNV_(other.fp_vkDestroyCudaModuleNV_),
      fp_vkDestroyCudaFunctionNV_(other.fp_vkDestroyCudaFunctionNV_),
      fp_vkCmdCudaLaunchKernelNV_(other.fp_vkCmdCudaLaunchKernelNV_) {
  other.device_ = VK_NULL_HANDLE;
  other.command_pool_ = VK_NULL_HANDLE;
}

VulkanDevice& VulkanDevice::operator=(VulkanDevice&& other) noexcept {
  if (this != &other) {
    cleanup();
    physical_device_ = other.physical_device_;
    device_ = other.device_;
    compute_queue_ = other.compute_queue_;
    compute_queue_family_ = other.compute_queue_family_;
    command_pool_ = other.command_pool_;
    info_ = std::move(other.info_);
    fp_vkCreateCudaModuleNV_ = other.fp_vkCreateCudaModuleNV_;
    fp_vkGetCudaModuleCacheNV_ = other.fp_vkGetCudaModuleCacheNV_;
    fp_vkCreateCudaFunctionNV_ = other.fp_vkCreateCudaFunctionNV_;
    fp_vkDestroyCudaModuleNV_ = other.fp_vkDestroyCudaModuleNV_;
    fp_vkDestroyCudaFunctionNV_ = other.fp_vkDestroyCudaFunctionNV_;
    fp_vkCmdCudaLaunchKernelNV_ = other.fp_vkCmdCudaLaunchKernelNV_;
    other.device_ = VK_NULL_HANDLE;
    other.command_pool_ = VK_NULL_HANDLE;
  }
  return *this;
}

void VulkanDevice::init_instance() { instance(); }

void VulkanDevice::select_physical_device(uint32_t device_index) {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  if (device_count == 0) {
    throw std::runtime_error("No Vulkan-capable GPUs found");
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

  if (device_index >= device_count) {
    throw std::runtime_error("Device index " + std::to_string(device_index) +
                             " out of range (found " +
                             std::to_string(device_count) + " devices)");
  }

  physical_device_ = devices[device_index];
  info_.device_id = device_index;
}

void VulkanDevice::create_device() {
  // Find a compute queue family
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_,
                                           &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_,
                                           &queue_family_count,
                                           queue_families.data());

  compute_queue_family_ = UINT32_MAX;
  for (uint32_t i = 0; i < queue_family_count; i++) {
    if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      compute_queue_family_ = i;
      break;
    }
  }
  if (compute_queue_family_ == UINT32_MAX) {
    throw std::runtime_error("No compute queue family found");
  }

  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_create_info = {};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = compute_queue_family_;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &queue_priority;

  // Check which extensions are available
  uint32_t ext_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count,
                                       nullptr);
  std::vector<VkExtensionProperties> available_exts(ext_count);
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &ext_count,
                                       available_exts.data());

  auto has_extension = [&](const char* name) {
    return std::any_of(available_exts.begin(), available_exts.end(),
                       [name](const VkExtensionProperties& ext) {
                         return strcmp(ext.extensionName, name) == 0;
                       });
  };

  // Required extensions
  std::vector<const char*> enabled_extensions;

  info_.cuda_kernel_launch_supported =
      has_extension(VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME);
  if (info_.cuda_kernel_launch_supported) {
    enabled_extensions.push_back(VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME);
  } else {
    throw std::runtime_error(
        "VK_NV_cuda_kernel_launch extension not available. "
        "Ensure you have a recent NVIDIA proprietary Vulkan driver.");
  }

  info_.buffer_device_address_supported =
      has_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  if (info_.buffer_device_address_supported) {
    enabled_extensions.push_back(
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  }

  // Enable features
  VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {};
  bda_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  bda_features.bufferDeviceAddress = VK_TRUE;

  VkPhysicalDeviceCudaKernelLaunchFeaturesNV cuda_features = {};
  cuda_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV;
  cuda_features.cudaKernelLaunchFeatures = VK_TRUE;
  cuda_features.pNext = &bda_features;

  VkPhysicalDeviceFeatures2 features2 = {};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &cuda_features;

  VkDeviceCreateInfo device_create_info = {};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.pNext = &features2;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pQueueCreateInfos = &queue_create_info;
  device_create_info.enabledExtensionCount =
      static_cast<uint32_t>(enabled_extensions.size());
  device_create_info.ppEnabledExtensionNames = enabled_extensions.data();

  VK_CHECK(vkCreateDevice(physical_device_, &device_create_info, nullptr,
                           &device_),
           "Failed to create Vulkan device");

  vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);

  // Load VK_NV_cuda_kernel_launch function pointers
  fp_vkCreateCudaModuleNV_ = reinterpret_cast<PFN_vkCreateCudaModuleNV>(
      vkGetDeviceProcAddr(device_, "vkCreateCudaModuleNV"));
  fp_vkGetCudaModuleCacheNV_ = reinterpret_cast<PFN_vkGetCudaModuleCacheNV>(
      vkGetDeviceProcAddr(device_, "vkGetCudaModuleCacheNV"));
  fp_vkCreateCudaFunctionNV_ = reinterpret_cast<PFN_vkCreateCudaFunctionNV>(
      vkGetDeviceProcAddr(device_, "vkCreateCudaFunctionNV"));
  fp_vkDestroyCudaModuleNV_ = reinterpret_cast<PFN_vkDestroyCudaModuleNV>(
      vkGetDeviceProcAddr(device_, "vkDestroyCudaModuleNV"));
  fp_vkDestroyCudaFunctionNV_ = reinterpret_cast<PFN_vkDestroyCudaFunctionNV>(
      vkGetDeviceProcAddr(device_, "vkDestroyCudaFunctionNV"));
  fp_vkCmdCudaLaunchKernelNV_ =
      reinterpret_cast<PFN_vkCmdCudaLaunchKernelNV>(
          vkGetDeviceProcAddr(device_, "vkCmdCudaLaunchKernelNV"));

  if (!fp_vkCreateCudaModuleNV_ || !fp_vkCmdCudaLaunchKernelNV_) {
    throw std::runtime_error(
        "Failed to load VK_NV_cuda_kernel_launch function pointers");
  }
}

void VulkanDevice::create_command_pool() {
  VkCommandPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = compute_queue_family_;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  VK_CHECK(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
           "Failed to create command pool");
}

void VulkanDevice::query_device_info() {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physical_device_, &props);
  info_.device_name = props.deviceName;

  // Query CUDA compute capability via VK_NV_cuda_kernel_launch properties
  VkPhysicalDeviceCudaKernelLaunchPropertiesNV cuda_props = {};
  cuda_props.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_PROPERTIES_NV;

  VkPhysicalDeviceProperties2 props2 = {};
  props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  props2.pNext = &cuda_props;
  vkGetPhysicalDeviceProperties2(physical_device_, &props2);

  info_.compute_capability_major = cuda_props.computeCapabilityMajor;
  info_.compute_capability_minor = cuda_props.computeCapabilityMinor;

  info_.max_compute_work_group_count[0] = props.limits.maxComputeWorkGroupCount[0];
  info_.max_compute_work_group_count[1] = props.limits.maxComputeWorkGroupCount[1];
  info_.max_compute_work_group_count[2] = props.limits.maxComputeWorkGroupCount[2];
  info_.max_compute_work_group_size[0] = props.limits.maxComputeWorkGroupSize[0];
  info_.max_compute_work_group_size[1] = props.limits.maxComputeWorkGroupSize[1];
  info_.max_compute_work_group_size[2] = props.limits.maxComputeWorkGroupSize[2];
  info_.max_compute_shared_memory_size = props.limits.maxComputeSharedMemorySize;

  // Query total memory
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
  info_.total_memory = 0;
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
    if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      info_.total_memory += mem_props.memoryHeaps[i].size;
    }
  }
}

bool VulkanDevice::is_cuda_kernel_launch_supported(uint32_t device_index) {
  VkInstance inst = instance();
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(inst, &device_count, nullptr);
  if (device_index >= device_count) return false;

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(inst, &device_count, devices.data());

  uint32_t ext_count = 0;
  vkEnumerateDeviceExtensionProperties(devices[device_index], nullptr,
                                       &ext_count, nullptr);
  std::vector<VkExtensionProperties> exts(ext_count);
  vkEnumerateDeviceExtensionProperties(devices[device_index], nullptr,
                                       &ext_count, exts.data());

  return std::any_of(exts.begin(), exts.end(),
                     [](const VkExtensionProperties& ext) {
                       return strcmp(ext.extensionName,
                                    VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME) == 0;
                     });
}

std::vector<DeviceInfo> VulkanDevice::enumerate_devices() {
  VkInstance inst = instance();
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(inst, &device_count, nullptr);

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(inst, &device_count, devices.data());

  std::vector<DeviceInfo> result;
  result.reserve(device_count);

  for (uint32_t i = 0; i < device_count; i++) {
    DeviceInfo info = {};
    info.device_id = i;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(devices[i], &props);
    info.device_name = props.deviceName;

    // Check extension support
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(devices[i], nullptr, &ext_count,
                                         nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(devices[i], nullptr, &ext_count,
                                         exts.data());

    info.cuda_kernel_launch_supported =
        std::any_of(exts.begin(), exts.end(),
                    [](const VkExtensionProperties& ext) {
                      return strcmp(ext.extensionName,
                                   VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME) == 0;
                    });

    info.buffer_device_address_supported =
        std::any_of(exts.begin(), exts.end(),
                    [](const VkExtensionProperties& ext) {
                      return strcmp(ext.extensionName,
                                   VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0;
                    });

    // Memory
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(devices[i], &mem_props);
    info.total_memory = 0;
    for (uint32_t j = 0; j < mem_props.memoryHeapCount; j++) {
      if (mem_props.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        info.total_memory += mem_props.memoryHeaps[j].size;
      }
    }

    result.push_back(std::move(info));
  }
  return result;
}

void VulkanDevice::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
    command_pool_ = VK_NULL_HANDLE;
  }
}

}  // namespace vulkan
}  // namespace vllm
