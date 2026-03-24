# SPDX-License-Identifier: Apache-2.0
# vLLM Vulkan Backend build configuration
#
# Builds the _vulkan_backend Python extension module.
# Requires: Vulkan SDK (headers + loader), pybind11
#
# Usage (standalone):
#   mkdir build && cd build
#   cmake -DVLLM_PYTHON_EXECUTABLE=$(which python3) ..
#   make -j$(nproc)
#
# Integrated into vLLM's main CMakeLists.txt:
#   add_subdirectory(csrc/vulkan)

cmake_minimum_required(VERSION 3.20)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Define the Vulkan backend module
pybind11_add_module(_vulkan_backend
    vulkan_device.cpp
    vulkan_memory.cpp
    vulkan_kernel.cpp
    vulkan_bindings.cpp
)

target_include_directories(_vulkan_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(_vulkan_backend PRIVATE
    ${Vulkan_LIBRARIES}
)

# Enable beta extensions (VK_NV_cuda_kernel_launch is provisional)
target_compile_definitions(_vulkan_backend PRIVATE
    VK_ENABLE_BETA_EXTENSIONS
)

target_compile_features(_vulkan_backend PRIVATE cxx_std_17)

# Install to vllm/vulkan/
install(TARGETS _vulkan_backend
    LIBRARY DESTINATION vllm/vulkan
    COMPONENT vulkan_backend
)
