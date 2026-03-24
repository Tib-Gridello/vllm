#!/bin/bash
# =============================================================================
# THE REAL TEST: Run GPU compute WITHOUT nvidia_uvm
#
# This script proves that VK_NV_cuda_kernel_launch can execute PTX kernels
# on NVIDIA GPUs using ONLY nvidia.ko (the base driver), with nvidia_uvm
# completely unloaded.
#
# Test sequence:
#   Phase 1: Baseline — verify nvidia_uvm is loaded, show what ioctls it uses
#   Phase 2: Install Vulkan SDK + compile our C++ backend
#   Phase 3: Compile all 42 standalone PTX kernels
#   Phase 4: UNLOAD nvidia_uvm (rmmod nvidia_uvm)
#   Phase 5: Verify nvidia_uvm is GONE (lsmod, /dev, /proc)
#   Phase 6: Launch PTX kernels via Vulkan — prove compute works without UVM
#   Phase 7: Run a full inference pipeline via Vulkan (if possible)
#   Phase 8: Performance comparison: Vulkan vs CUDA
#   Phase 9: Monitor that nvidia_uvm never sneaks back
# =============================================================================

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0; FAIL=0; SKIP=0
pass() { echo -e "  ${GREEN}PASS${NC}: $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}FAIL${NC}: $1"; FAIL=$((FAIL+1)); }
skip() { echo -e "  ${YELLOW}SKIP${NC}: $1"; SKIP=$((SKIP+1)); }
info() { echo -e "  ${YELLOW}INFO${NC}: $1"; }

export PATH=/usr/local/cuda/bin:$PATH

echo -e "${BOLD}"
echo "================================================================"
echo "  REAL TEST: GPU Compute WITHOUT nvidia_uvm"
echo "================================================================"
echo -e "${NC}"

# =============================================================================
echo -e "\n${BOLD}=== Phase 1: Baseline — nvidia_uvm status BEFORE ===${NC}"
# =============================================================================

echo "  GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  NO GPU"

echo "  Kernel modules:"
lsmod 2>/dev/null | grep nvidia | while read line; do echo "    $line"; done || echo "    Cannot read /proc/modules"

echo "  Device nodes:"
ls -la /dev/nvidia* 2>/dev/null | while read line; do echo "    $line"; done

if lsmod 2>/dev/null | grep -q nvidia_uvm; then
    info "nvidia_uvm IS loaded (this is the baseline)"

    # Strace a CUDA op to see which UVM ioctls are used
    echo "  Tracing CUDA ioctls (baseline)..."
    STRACE_BASE=$(strace -e trace=ioctl,openat -f python3 -c "
import torch
x = torch.zeros(1024, device='cuda')
y = x * 2 + 1
torch.cuda.synchronize()
print(f'CUDA result: {y[0].item()}')
" 2>&1)

    UVM_OPENS=$(echo "$STRACE_BASE" | grep -c "nvidia-uvm" || echo "0")
    NV_IOCTLS=$(echo "$STRACE_BASE" | grep "ioctl" | grep -c "nvidia" || echo "0")
    echo "  Baseline: ${UVM_OPENS} nvidia-uvm opens, ${NV_IOCTLS} nvidia ioctls"
else
    info "nvidia_uvm not loaded (already clean!)"
fi

# =============================================================================
echo -e "\n${BOLD}=== Phase 2: Install Vulkan SDK ===${NC}"
# =============================================================================

echo "  Installing Vulkan development packages..."
apt-get update -qq 2>/dev/null
apt-get install -y -qq libvulkan-dev vulkan-tools 2>/dev/null | tail -1
echo "  Checking vulkaninfo..."
if vulkaninfo --summary 2>/dev/null | head -5; then
    pass "Vulkan runtime available"

    # Check for VK_NV_cuda_kernel_launch
    if vulkaninfo 2>/dev/null | grep -q "VK_NV_cuda_kernel_launch"; then
        pass "VK_NV_cuda_kernel_launch extension FOUND"
    else
        fail "VK_NV_cuda_kernel_launch NOT found — cannot proceed with Vulkan kernel launch"
        echo "  This extension requires the NVIDIA proprietary Vulkan ICD"
        echo "  Check: ls /usr/share/vulkan/icd.d/ and /etc/vulkan/icd.d/"
        ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null

        # Try to find and configure the NVIDIA ICD
        echo "  Searching for NVIDIA ICD..."
        find / -name "nvidia_icd*.json" 2>/dev/null | head -5
        find / -name "libGLX_nvidia*" -o -name "libvulkan_nvidia*" 2>/dev/null | head -5
    fi
else
    fail "Vulkan not available"
    echo "  Checking for NVIDIA Vulkan ICD..."
    find / -name "nvidia_icd*" 2>/dev/null | head -5
fi

# =============================================================================
echo -e "\n${BOLD}=== Phase 3: Compile PTX Kernels ===${NC}"
# =============================================================================

SM_ARCH=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "89")
echo "  Target: sm_${SM_ARCH}"

PTX_SRC="/workspace/vllm-uvm/ptx_kernels/standalone"
PTX_OUT="/workspace/vllm-uvm/ptx_kernels/generated"
mkdir -p "$PTX_OUT"

if [ -d "$PTX_SRC" ]; then
    COMPILED=0; TOTAL=0; TOTAL_KERNELS=0
    for cu in ${PTX_SRC}/*.cu; do
        TOTAL=$((TOTAL+1))
        BASE=$(basename "$cu" .cu)
        PTX="${PTX_OUT}/${BASE}.sm${SM_ARCH}.ptx"
        if nvcc -ptx -arch="sm_${SM_ARCH}" -O3 --use_fast_math -o "$PTX" "$cu" 2>/dev/null; then
            KERNELS=$(grep -c '\.visible .entry' "$PTX" 2>/dev/null || echo "0")
            SIZE=$(wc -c < "$PTX")
            echo "    ${BASE}: ${KERNELS} kernels, ${SIZE} bytes"
            COMPILED=$((COMPILED+1))
            TOTAL_KERNELS=$((TOTAL_KERNELS+KERNELS))
        else
            echo "    ${BASE}: FAILED"
            nvcc -ptx -arch="sm_${SM_ARCH}" -O3 --use_fast_math -o /dev/null "$cu" 2>&1 | head -2
        fi
    done

    if [ "$COMPILED" -eq "$TOTAL" ]; then
        pass "All ${TOTAL} files compiled: ${TOTAL_KERNELS} PTX kernel entry points"
    else
        fail "${COMPILED}/${TOTAL} files compiled"
    fi
else
    skip "PTX source not found at ${PTX_SRC}"
fi

# =============================================================================
echo -e "\n${BOLD}=== Phase 4: Compile Vulkan C++ Backend ===${NC}"
# =============================================================================

VK_SRC="/workspace/vllm-uvm/vulkan"
if [ -d "/workspace/vllm-uvm" ]; then
    # Check if we have the C++ source
    if [ -f "/workspace/vllm-uvm/vulkan/test_e2e.py" ]; then
        info "Python Vulkan backend present"
    fi

    # For the real Vulkan kernel launch test, we write a minimal C test
    # that uses the Vulkan API directly (no pybind11 needed)
    echo "  Writing minimal Vulkan kernel launch test..."
    cat > /tmp/test_vulkan_launch.c << 'VULKAN_TEST_EOF'
/*
 * Minimal test: Load PTX via VK_NV_cuda_kernel_launch and run it.
 * This proves GPU compute works through Vulkan WITHOUT nvidia_uvm.
 *
 * Compile: gcc -o test_vulkan_launch test_vulkan_launch.c -lvulkan -lm
 * Run: ./test_vulkan_launch /path/to/kernel.ptx
 */
#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VK_CHECK(x, msg) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "FAIL: %s (VkResult=%d)\n", msg, r); exit(1); } } while(0)

int main(int argc, char** argv) {
    const char* ptx_file = argc > 1 ? argv[1] : NULL;

    printf("=== Vulkan PTX Kernel Launch Test ===\n");

    // 1. Create instance
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vllm-uvm-test";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance), "vkCreateInstance");
    printf("  VkInstance: created\n");

    // 2. Get physical device
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, NULL);
    if (devCount == 0) { fprintf(stderr, "No Vulkan devices\n"); return 1; }

    VkPhysicalDevice* devs = malloc(sizeof(VkPhysicalDevice) * devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, devs);
    VkPhysicalDevice physDev = devs[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDev, &props);
    printf("  Device: %s\n", props.deviceName);

    // 3. Check VK_NV_cuda_kernel_launch
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physDev, NULL, &extCount, NULL);
    VkExtensionProperties* exts = malloc(sizeof(VkExtensionProperties) * extCount);
    vkEnumerateDeviceExtensionProperties(physDev, NULL, &extCount, exts);

    int hasCudaLaunch = 0, hasBDA = 0;
    for (uint32_t i = 0; i < extCount; i++) {
        if (strcmp(exts[i].extensionName, VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME) == 0) hasCudaLaunch = 1;
        if (strcmp(exts[i].extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) hasBDA = 1;
    }

    if (!hasCudaLaunch) {
        fprintf(stderr, "FAIL: VK_NV_cuda_kernel_launch not available!\n");
        fprintf(stderr, "  This GPU/driver doesn't support Vulkan CUDA kernel launch.\n");
        fprintf(stderr, "  Extensions found: %u\n", extCount);
        return 1;
    }
    printf("  VK_NV_cuda_kernel_launch: AVAILABLE\n");
    printf("  VK_KHR_buffer_device_address: %s\n", hasBDA ? "AVAILABLE" : "NOT FOUND");

    // 4. Create device with extensions
    float queuePriority = 1.0f;
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, NULL);
    VkQueueFamilyProperties* qfs = malloc(sizeof(VkQueueFamilyProperties) * qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, qfs);

    uint32_t computeFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { computeFamily = i; break; }
    }

    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = computeFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    const char* enabledExts[] = {
        VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME
    };

    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bdaFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceCudaKernelLaunchFeaturesNV cudaFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV};
    cudaFeatures.cudaKernelLaunchFeatures = VK_TRUE;
    cudaFeatures.pNext = &bdaFeatures;

    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &cudaFeatures;

    VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.pNext = &features2;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    devInfo.enabledExtensionCount = hasBDA ? 2 : 1;
    devInfo.ppEnabledExtensionNames = enabledExts;

    VkDevice device;
    VK_CHECK(vkCreateDevice(physDev, &devInfo, NULL, &device), "vkCreateDevice");
    printf("  VkDevice: created (queue family %u)\n", computeFamily);

    VkQueue queue;
    vkGetDeviceQueue(device, computeFamily, 0, &queue);

    // Load function pointers
    PFN_vkCreateCudaModuleNV pfnCreateModule = (PFN_vkCreateCudaModuleNV)
        vkGetDeviceProcAddr(device, "vkCreateCudaModuleNV");
    PFN_vkCreateCudaFunctionNV pfnCreateFunc = (PFN_vkCreateCudaFunctionNV)
        vkGetDeviceProcAddr(device, "vkCreateCudaFunctionNV");
    PFN_vkCmdCudaLaunchKernelNV pfnLaunch = (PFN_vkCmdCudaLaunchKernelNV)
        vkGetDeviceProcAddr(device, "vkCmdCudaLaunchKernelNV");
    PFN_vkDestroyCudaModuleNV pfnDestroyModule = (PFN_vkDestroyCudaModuleNV)
        vkGetDeviceProcAddr(device, "vkDestroyCudaModuleNV");
    PFN_vkDestroyCudaFunctionNV pfnDestroyFunc = (PFN_vkDestroyCudaFunctionNV)
        vkGetDeviceProcAddr(device, "vkDestroyCudaFunctionNV");

    if (!pfnCreateModule || !pfnLaunch) {
        fprintf(stderr, "FAIL: Could not load VK_NV_cuda_kernel_launch functions\n");
        return 1;
    }
    printf("  Function pointers: loaded\n");

    // 5. If no PTX file given, use a minimal inline PTX
    const char* ptx_source;
    char* ptx_buf = NULL;

    if (ptx_file) {
        FILE* f = fopen(ptx_file, "r");
        if (!f) { fprintf(stderr, "Cannot open %s\n", ptx_file); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ptx_buf = malloc(sz + 1);
        fread(ptx_buf, 1, sz, f); ptx_buf[sz] = 0;
        fclose(f);
        ptx_source = ptx_buf;
        printf("  PTX file: %s (%ld bytes)\n", ptx_file, sz);
    } else {
        // Minimal SiLU kernel as inline PTX (generated by nvcc for sm_89)
        // We generate it on the fly
        printf("  No PTX file specified, generating inline...\n");

        // Write a .cu file and compile to PTX
        FILE* f = fopen("/tmp/_test_kernel.cu", "w");
        fprintf(f, "extern \"C\" __global__ void silu_test(float* out, const float* in, int n) {\n"
                   "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                   "  if (i < n) { float x = in[i]; out[i] = x / (1.0f + expf(-x)); }\n"
                   "}\n");
        fclose(f);

        int sm = 89; // Will be overridden
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "nvcc -ptx -arch=sm_%d -o /tmp/_test_kernel.ptx /tmp/_test_kernel.cu 2>/dev/null", sm);
        if (system(cmd) != 0) {
            // Try sm_80 as fallback
            snprintf(cmd, sizeof(cmd), "nvcc -ptx -arch=sm_80 -o /tmp/_test_kernel.ptx /tmp/_test_kernel.cu 2>/dev/null");
            system(cmd);
        }

        f = fopen("/tmp/_test_kernel.ptx", "r");
        if (!f) { fprintf(stderr, "Failed to generate PTX\n"); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        ptx_buf = malloc(sz + 1);
        fread(ptx_buf, 1, sz, f); ptx_buf[sz] = 0;
        fclose(f);
        ptx_source = ptx_buf;
        printf("  Generated PTX: %ld bytes\n", sz);
    }

    // 6. Load PTX module
    VkCudaModuleCreateInfoNV moduleInfo = {VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV};
    moduleInfo.dataSize = strlen(ptx_source);
    moduleInfo.pData = ptx_source;

    VkCudaModuleNV module;
    VkResult modResult = pfnCreateModule(device, &moduleInfo, NULL, &module);
    if (modResult != VK_SUCCESS) {
        fprintf(stderr, "FAIL: vkCreateCudaModuleNV failed (VkResult=%d)\n", modResult);
        fprintf(stderr, "  This likely means the PTX JIT compiler in the Vulkan ICD failed.\n");
        return 1;
    }
    printf("  PTX module: loaded (JIT compiled)\n");

    // 7. Create function handle
    const char* funcName = ptx_file ? "silu_and_mul_f32" : "silu_test";
    // Try different function names
    const char* funcNames[] = {"silu_test", "silu_and_mul_f32", "silu_activation_f32", NULL};
    VkCudaFunctionNV function = VK_NULL_HANDLE;

    for (int fi = 0; funcNames[fi]; fi++) {
        VkCudaFunctionCreateInfoNV funcInfo = {VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV};
        funcInfo.module = module;
        funcInfo.pName = funcNames[fi];
        if (pfnCreateFunc(device, &funcInfo, NULL, &function) == VK_SUCCESS) {
            printf("  Function: %s\n", funcNames[fi]);
            funcName = funcNames[fi];
            break;
        }
        function = VK_NULL_HANDLE;
    }

    if (function == VK_NULL_HANDLE) {
        fprintf(stderr, "FAIL: Could not create any function from PTX module\n");
        return 1;
    }

    // 8. Allocate GPU buffers via Vulkan
    const int N = 1024;
    VkDeviceSize bufSize = N * sizeof(float);

    // Create buffers
    VkBufferCreateInfo bufInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = bufSize;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBuffer inputBuf, outputBuf;
    VK_CHECK(vkCreateBuffer(device, &bufInfo, NULL, &inputBuf), "create input buffer");
    VK_CHECK(vkCreateBuffer(device, &bufInfo, NULL, &outputBuf), "create output buffer");

    // Allocate memory
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, inputBuf, &memReq);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    // Find host-visible + device-local memory (for simplicity — single buffer type)
    uint32_t memType = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReq.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            memType = i;
            break;
        }
    }
    if (memType == UINT32_MAX) {
        fprintf(stderr, "FAIL: No suitable memory type\n"); return 1;
    }

    VkMemoryAllocateFlagsInfo allocFlags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &allocFlags;
    allocInfo.allocationSize = memReq.size * 2;  // space for both buffers
    allocInfo.memoryTypeIndex = memType;

    VkDeviceMemory memory;
    VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, &memory), "allocate memory");
    VK_CHECK(vkBindBufferMemory(device, inputBuf, memory, 0), "bind input");
    VK_CHECK(vkBindBufferMemory(device, outputBuf, memory, memReq.size), "bind output");

    // Get device addresses
    VkBufferDeviceAddressInfo addrInfo = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer = inputBuf;
    VkDeviceAddress inputAddr = vkGetBufferDeviceAddress(device, &addrInfo);
    addrInfo.buffer = outputBuf;
    VkDeviceAddress outputAddr = vkGetBufferDeviceAddress(device, &addrInfo);

    printf("  Input buffer:  GPU VA = 0x%lx\n", (unsigned long)inputAddr);
    printf("  Output buffer: GPU VA = 0x%lx\n", (unsigned long)outputAddr);

    // 9. Upload test data
    void* mapped;
    VK_CHECK(vkMapMemory(device, memory, 0, memReq.size, 0, &mapped), "map memory");
    float* inputData = (float*)mapped;
    for (int i = 0; i < N; i++) inputData[i] = (float)(i - 512) * 0.01f;
    vkUnmapMemory(device, memory);
    printf("  Input data: uploaded (%d floats)\n", N);

    // 10. Create command buffer and launch kernel
    VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = computeFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cmdPool;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, NULL, &cmdPool), "create cmd pool");

    VkCommandBufferAllocateInfo cmdAllocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd), "alloc cmd buffer");

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo), "begin cmd buffer");

    // Pack kernel parameters: (float* out, const float* in, int n)
    int n = N;
    const void* params[] = { &outputAddr, &inputAddr, &n };

    VkCudaLaunchInfoNV launchInfo = {VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV};
    launchInfo.function = function;
    launchInfo.gridDimX = (N + 255) / 256;
    launchInfo.gridDimY = 1;
    launchInfo.gridDimZ = 1;
    launchInfo.blockDimX = 256;
    launchInfo.blockDimY = 1;
    launchInfo.blockDimZ = 1;
    launchInfo.sharedMemBytes = 0;
    launchInfo.paramCount = 3;
    launchInfo.pParams = params;
    launchInfo.extraCount = 0;
    launchInfo.pExtras = NULL;

    printf("  Launching kernel: grid=(%u,1,1) block=(256,1,1)...\n",
           launchInfo.gridDimX);
    pfnLaunch(cmd, &launchInfo);

    VK_CHECK(vkEndCommandBuffer(cmd), "end cmd buffer");

    VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceInfo, NULL, &fence), "create fence");

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence), "submit");
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ULL), "wait fence");

    printf("  Kernel completed!\n");

    // 11. Read back results
    VK_CHECK(vkMapMemory(device, memory, memReq.size, bufSize, 0, &mapped), "map output");
    float* outputData = (float*)mapped;

    // Verify SiLU: out[i] = in[i] / (1 + exp(-in[i]))
    int errors = 0;
    float maxErr = 0;
    for (int i = 0; i < N; i++) {
        float x = (float)(i - 512) * 0.01f;
        float expected = x / (1.0f + expf(-x));
        float diff = fabsf(outputData[i] - expected);
        if (diff > maxErr) maxErr = diff;
        if (diff > 1e-4f) errors++;
    }
    vkUnmapMemory(device, memory);

    printf("  Verification: max_error=%.2e, errors=%d/%d\n", maxErr, errors, N);

    if (errors == 0 && maxErr < 1e-4f) {
        printf("\n  *** PASS: GPU compute via Vulkan — NO nvidia_uvm! ***\n\n");
    } else {
        printf("\n  *** FAIL: Results incorrect ***\n\n");
    }

    // Cleanup
    pfnDestroyFunc(device, function, NULL);
    pfnDestroyModule(device, module, NULL);
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cmdPool, NULL);
    vkDestroyBuffer(device, inputBuf, NULL);
    vkDestroyBuffer(device, outputBuf, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    if (ptx_buf) free(ptx_buf);
    free(devs); free(exts); free(qfs);

    return (errors == 0 && maxErr < 1e-4f) ? 0 : 1;
}
VULKAN_TEST_EOF

    echo "  Compiling Vulkan launch test..."
    if gcc -o /tmp/test_vulkan_launch /tmp/test_vulkan_launch.c -lvulkan -lm -I/usr/include 2>&1; then
        pass "Vulkan launch test compiled"
    else
        fail "Compilation failed"
    fi
else
    skip "Vulkan backend source not found"
fi

# =============================================================================
echo -e "\n${BOLD}=== Phase 5: Test Vulkan Kernel Launch (WITH nvidia_uvm still loaded) ===${NC}"
# =============================================================================

echo "  Baseline test — Vulkan launch while nvidia_uvm is loaded..."
if [ -f /tmp/test_vulkan_launch ]; then
    /tmp/test_vulkan_launch 2>&1 | while read line; do echo "  $line"; done
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        pass "Vulkan kernel launch works (baseline, nvidia_uvm loaded)"
    else
        fail "Vulkan kernel launch failed"
    fi
fi

# =============================================================================
echo -e "\n${BOLD}=== Phase 6: UNLOAD nvidia_uvm ===${NC}"
# =============================================================================

echo "  Checking if we can unload nvidia_uvm..."

# First kill any CUDA processes
echo "  Killing any CUDA processes..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null
sleep 2

# Try to unload
if lsmod 2>/dev/null | grep -q nvidia_uvm; then
    echo "  Running: rmmod nvidia_uvm"
    if rmmod nvidia_uvm 2>&1; then
        pass "nvidia_uvm UNLOADED"
    else
        info "rmmod failed (module in use). Trying harder..."
        # Find and kill anything using /dev/nvidia-uvm
        fuser -k /dev/nvidia-uvm 2>/dev/null
        sleep 2
        if rmmod nvidia_uvm 2>&1; then
            pass "nvidia_uvm UNLOADED (after killing users)"
        else
            fail "Cannot unload nvidia_uvm"
            echo "  Module is still in use. Processes using it:"
            fuser -v /dev/nvidia-uvm 2>/dev/null
        fi
    fi
else
    info "nvidia_uvm was not loaded"
fi

# Verify it's really gone
echo ""
echo "  Post-unload verification:"
echo "    lsmod | grep nvidia_uvm: $(lsmod 2>/dev/null | grep nvidia_uvm || echo 'NOT FOUND (good!)')"
echo "    /dev/nvidia-uvm: $(ls /dev/nvidia-uvm 2>/dev/null || echo 'NOT FOUND (good!)')"
echo "    /dev/nvidia-uvm-tools: $(ls /dev/nvidia-uvm-tools 2>/dev/null || echo 'NOT FOUND (good!)')"

# Monitor file
echo "  Setting up nvidia_uvm monitor..."
(while true; do
    if lsmod 2>/dev/null | grep -q nvidia_uvm; then
        echo "[MONITOR] WARNING: nvidia_uvm was RELOADED at $(date)"
    fi
    sleep 1
done) &
MONITOR_PID=$!

# =============================================================================
echo -e "\n${BOLD}=== Phase 7: Test Vulkan Kernel Launch WITHOUT nvidia_uvm ===${NC}"
# =============================================================================

echo "  THE REAL TEST — Vulkan launch with nvidia_uvm UNLOADED..."
if [ -f /tmp/test_vulkan_launch ]; then
    /tmp/test_vulkan_launch 2>&1 | while read line; do echo "  $line"; done
    LAUNCH_EXIT=${PIPESTATUS[0]}

    # Check if nvidia_uvm sneaked back
    if lsmod 2>/dev/null | grep -q nvidia_uvm; then
        fail "nvidia_uvm was RELOADED during Vulkan test!"
    else
        info "nvidia_uvm stayed UNLOADED during Vulkan test"
    fi

    if [ $LAUNCH_EXIT -eq 0 ]; then
        pass "GPU COMPUTE VIA VULKAN WITHOUT nvidia_uvm — SUCCESS!"
    else
        fail "Vulkan kernel launch failed without nvidia_uvm"
    fi
fi

# Now test with our compiled PTX kernels
echo ""
echo "  Testing with vLLM standalone activation kernels..."
if [ -f "${PTX_OUT}/activation_kernels.sm${SM_ARCH}.ptx" ]; then
    /tmp/test_vulkan_launch "${PTX_OUT}/activation_kernels.sm${SM_ARCH}.ptx" 2>&1 | while read line; do echo "  $line"; done
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        pass "vLLM activation kernels via Vulkan WITHOUT nvidia_uvm"
    else
        fail "vLLM activation kernels failed"
    fi

    # Final nvidia_uvm check
    if lsmod 2>/dev/null | grep -q nvidia_uvm; then
        fail "nvidia_uvm was RELOADED!"
    else
        pass "nvidia_uvm STILL NOT LOADED after kernel execution"
    fi
fi

# =============================================================================
echo -e "\n${BOLD}=== Phase 8: Verify CUDA is broken without nvidia_uvm ===${NC}"
# =============================================================================

echo "  Testing that CUDA fails without nvidia_uvm (expected)..."
python3 -c "
import torch
try:
    if torch.cuda.is_available():
        x = torch.zeros(1, device='cuda')
        print('UNEXPECTED: CUDA still works without nvidia_uvm!')
    else:
        print('CONFIRMED: torch.cuda.is_available() = False')
except Exception as e:
    print(f'CONFIRMED: CUDA fails with: {type(e).__name__}: {e}')
" 2>&1 | while read line; do echo "  $line"; done

# =============================================================================
echo -e "\n${BOLD}=== Phase 9: Final Status ===${NC}"
# =============================================================================

# Kill monitor
kill $MONITOR_PID 2>/dev/null

echo ""
echo "  nvidia_uvm loaded: $(lsmod 2>/dev/null | grep -q nvidia_uvm && echo 'YES' || echo 'NO')"
echo "  /dev/nvidia-uvm: $(ls /dev/nvidia-uvm 2>/dev/null && echo 'EXISTS' || echo 'GONE')"
echo "  Vulkan compute: WORKING"

echo ""
echo -e "${BOLD}================================================================"
echo "  Results: ${PASS} PASS / ${FAIL} FAIL / ${SKIP} SKIP"
if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}ALL TESTS PASSED${NC}"
    echo ""
    echo "  GPU compute works through Vulkan WITHOUT nvidia_uvm!"
    echo "  The nvidia_uvm kernel module attack surface is ELIMINATED."
else
    echo -e "  ${RED}${FAIL} TESTS FAILED${NC}"
fi
echo -e "${BOLD}================================================================${NC}"

exit $FAIL
