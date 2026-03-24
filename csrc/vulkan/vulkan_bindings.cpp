// SPDX-License-Identifier: Apache-2.0
// vLLM Vulkan Backend - Python bindings via pybind11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vulkan_device.h"
#include "vulkan_kernel.h"
#include "vulkan_memory.h"

namespace py = pybind11;
using namespace vllm::vulkan;

PYBIND11_MODULE(_vulkan_backend, m) {
  m.doc() = "vLLM Vulkan backend — GPU compute without nvidia_uvm";

  // DeviceInfo
  py::class_<DeviceInfo>(m, "DeviceInfo")
      .def_readonly("device_id", &DeviceInfo::device_id)
      .def_readonly("device_name", &DeviceInfo::device_name)
      .def_readonly("compute_capability_major",
                    &DeviceInfo::compute_capability_major)
      .def_readonly("compute_capability_minor",
                    &DeviceInfo::compute_capability_minor)
      .def_readonly("total_memory", &DeviceInfo::total_memory)
      .def_readonly("cuda_kernel_launch_supported",
                    &DeviceInfo::cuda_kernel_launch_supported)
      .def_readonly("buffer_device_address_supported",
                    &DeviceInfo::buffer_device_address_supported)
      .def("__repr__", [](const DeviceInfo& info) {
        return "<DeviceInfo '" + info.device_name + "' sm_" +
               std::to_string(info.compute_capability_major) +
               std::to_string(info.compute_capability_minor) + " " +
               std::to_string(info.total_memory / (1024 * 1024)) + "MB" +
               (info.cuda_kernel_launch_supported ? " cuda_launch" : "") + ">";
      });

  // VulkanDevice
  py::class_<VulkanDevice>(m, "VulkanDevice")
      .def(py::init<uint32_t>(), py::arg("device_index") = 0)
      .def("info", &VulkanDevice::info, py::return_value_policy::reference)
      .def_static("is_cuda_kernel_launch_supported",
                  &VulkanDevice::is_cuda_kernel_launch_supported,
                  py::arg("device_index") = 0)
      .def_static("enumerate_devices", &VulkanDevice::enumerate_devices);

  // VulkanBuffer
  py::class_<VulkanBuffer>(m, "VulkanBuffer")
      .def_readonly("device_address", &VulkanBuffer::device_address)
      .def_readonly("size", &VulkanBuffer::size)
      .def("__repr__", [](const VulkanBuffer& buf) {
        return "<VulkanBuffer addr=0x" +
               ([&]() {
                 char hex[32];
                 snprintf(hex, sizeof(hex), "%llx",
                          (unsigned long long)buf.device_address);
                 return std::string(hex);
               })() +
               " size=" + std::to_string(buf.size) + ">";
      });

  // VulkanMemoryAllocator
  py::class_<VulkanMemoryAllocator>(m, "VulkanMemoryAllocator")
      .def(py::init<VulkanDevice&>())
      .def("allocate_device", &VulkanMemoryAllocator::allocate_device,
           py::arg("size"))
      .def("allocate_staging", &VulkanMemoryAllocator::allocate_staging,
           py::arg("size"))
      .def("free", &VulkanMemoryAllocator::free)
      .def("upload",
           [](VulkanMemoryAllocator& alloc, py::bytes data,
              VulkanBuffer& dst) {
             std::string s = data;
             alloc.upload(s.data(), dst, s.size());
           })
      .def("download",
           [](VulkanMemoryAllocator& alloc, const VulkanBuffer& src,
              size_t size) {
             std::vector<uint8_t> buf(size);
             alloc.download(src, buf.data(), size);
             return py::bytes(reinterpret_cast<const char*>(buf.data()), size);
           })
      .def("copy", &VulkanMemoryAllocator::copy)
      .def("total_allocated", &VulkanMemoryAllocator::total_allocated);

  // KernelModule
  py::class_<KernelModule>(m, "KernelModule")
      .def(py::init<VulkanDevice&, const std::string&>(), py::arg("device"),
           py::arg("ptx_source"),
           "Load a CUDA module from PTX source code")
      .def(py::init([](VulkanDevice& device, py::bytes cache_data) {
             std::string s = cache_data;
             return KernelModule(device, s.data(), s.size());
           }),
           py::arg("device"), py::arg("cache_data"),
           "Load a CUDA module from a binary cache (skips JIT compilation)")
      .def("get_binary_cache",
           [](const KernelModule& m) {
             auto cache = m.get_binary_cache();
             return py::bytes(reinterpret_cast<const char*>(cache.data()),
                              cache.size());
           },
           "Get compiled binary cache for fast reload (returns bytes)");

  // KernelFunction
  py::class_<KernelFunction>(m, "KernelFunction")
      .def(py::init<VulkanDevice&, KernelModule&, const std::string&>(),
           py::arg("device"), py::arg("module"), py::arg("function_name"))
      .def("name", &KernelFunction::name);

  // LaunchConfig
  py::class_<LaunchConfig>(m, "LaunchConfig")
      .def(py::init<>())
      .def_readwrite("grid_x", &LaunchConfig::grid_x)
      .def_readwrite("grid_y", &LaunchConfig::grid_y)
      .def_readwrite("grid_z", &LaunchConfig::grid_z)
      .def_readwrite("block_x", &LaunchConfig::block_x)
      .def_readwrite("block_y", &LaunchConfig::block_y)
      .def_readwrite("block_z", &LaunchConfig::block_z)
      .def_readwrite("shared_mem_bytes", &LaunchConfig::shared_mem_bytes);

  // KernelLauncher
  py::class_<KernelLauncher>(m, "KernelLauncher")
      .def(py::init<VulkanDevice&>())
      .def(
          "launch_sync",
          [](KernelLauncher& launcher, const KernelFunction& func,
             const LaunchConfig& config, py::list params) {
            // Convert Python list of parameter values to void* array.
            // Each element should be a bytes object containing the raw
            // parameter value (e.g., struct.pack('<Q', device_address)
            // for a pointer, struct.pack('<i', n) for an int).
            std::vector<std::string> param_storage;
            std::vector<const void*> param_ptrs;
            param_storage.reserve(params.size());
            param_ptrs.reserve(params.size());

            for (auto& p : params) {
              param_storage.push_back(p.cast<std::string>());
              param_ptrs.push_back(param_storage.back().data());
            }

            launcher.launch_sync(func, config, param_ptrs.data(),
                                 param_ptrs.size());
          },
          py::arg("func"), py::arg("config"), py::arg("params"),
          "Launch a kernel synchronously. params is a list of bytes objects, "
          "each containing the raw parameter value (use struct.pack).")
      .def("begin_recording", &KernelLauncher::begin_recording)
      .def(
          "record_launch",
          [](KernelLauncher& launcher, const KernelFunction& func,
             const LaunchConfig& config, py::list params) {
            std::vector<std::string> param_storage;
            std::vector<const void*> param_ptrs;
            param_storage.reserve(params.size());
            param_ptrs.reserve(params.size());

            for (auto& p : params) {
              param_storage.push_back(p.cast<std::string>());
              param_ptrs.push_back(param_storage.back().data());
            }

            launcher.record_launch(func, config, param_ptrs.data(),
                                   param_ptrs.size());
          },
          py::arg("func"), py::arg("config"), py::arg("params"))
      .def("submit_and_wait", &KernelLauncher::submit_and_wait)
      .def("submit", &KernelLauncher::submit)
      .def("wait", &KernelLauncher::wait);

  // Module-level functions
  m.def("is_available", []() {
    try {
      auto devices = VulkanDevice::enumerate_devices();
      for (auto& d : devices) {
        if (d.cuda_kernel_launch_supported) return true;
      }
      return false;
    } catch (...) {
      return false;
    }
  }, "Check if Vulkan backend with VK_NV_cuda_kernel_launch is available");
}
