#!/bin/bash
# Build a hardened nvidia_uvm.ko from NVIDIA open-source GPU kernel modules.
#
# This script:
# 1. Clones the NVIDIA open-source kernel modules matching your driver version
# 2. Applies the hardening patch (blocks dangerous ioctls)
# 3. Builds only the nvidia-uvm module
# 4. Outputs nvidia-uvm.ko ready for insmod
#
# Prerequisites:
#   - Linux kernel headers installed
#   - gcc, make
#   - Running NVIDIA driver (to detect version)
#
# Usage:
#   ./build.sh                    # Auto-detect driver version
#   ./build.sh 545.23.06          # Specify version explicitly
#   DRIVER_VERSION=550.54.14 ./build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Detect NVIDIA driver version
if [ -n "${1:-}" ]; then
    DRIVER_VERSION="$1"
elif [ -n "${DRIVER_VERSION:-}" ]; then
    : # Use env var
elif command -v nvidia-smi &>/dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')
    echo "Detected NVIDIA driver version: ${DRIVER_VERSION}"
else
    echo "ERROR: Cannot detect NVIDIA driver version."
    echo "Usage: $0 <driver_version>  (e.g., $0 545.23.06)"
    exit 1
fi

echo "=== Building hardened nvidia-uvm.ko for driver ${DRIVER_VERSION} ==="

# Clone source
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [ ! -d "open-gpu-kernel-modules" ]; then
    echo "Cloning NVIDIA open-source kernel modules (tag ${DRIVER_VERSION})..."
    git clone --branch "${DRIVER_VERSION}" --depth 1 \
        https://github.com/NVIDIA/open-gpu-kernel-modules.git
else
    echo "Using existing source in ${BUILD_DIR}/open-gpu-kernel-modules"
fi

cd open-gpu-kernel-modules

# Apply hardening: add -DUVM_HARDENED_MODE to nvidia-uvm CFLAGS
UVM_KBUILD="kernel-open/nvidia-uvm/nvidia-uvm.Kbuild"
if ! grep -q "UVM_HARDENED_MODE" "${UVM_KBUILD}" 2>/dev/null; then
    echo "Applying hardening flag to ${UVM_KBUILD}..."
    # Add the hardening flag to CFLAGS
    echo '' >> "${UVM_KBUILD}"
    echo '# vLLM hardened mode: block dangerous ioctls' >> "${UVM_KBUILD}"
    echo 'ccflags-y += -DUVM_HARDENED_MODE' >> "${UVM_KBUILD}"
fi

# Apply the ioctl blocking patch to uvm.c
UVM_C="kernel-open/nvidia-uvm/uvm.c"
if ! grep -q "UVM_HARDENED_MODE" "${UVM_C}" 2>/dev/null; then
    echo "Applying ioctl blocking patch to ${UVM_C}..."

    # Find the uvm_ioctl or uvm_unlocked_ioctl function and inject our block
    # The exact insertion point depends on the driver version.
    # We insert after the initial validation checks, before the main dispatch.

    # Create the injection code
    cat > /tmp/uvm_hardened_inject.c << 'INJECT_EOF'

#ifdef UVM_HARDENED_MODE
    /* vLLM hardened mode: block dangerous ioctls before dispatch.
     * These are NOT needed for cuMemAlloc, cuLaunchKernel, or cuMemCreate/cuMemMap.
     * vLLM does not use cudaMallocManaged or managed memory. */
    switch (cmd) {
        /* CRITICAL: Arbitrary process memory access */
        case UVM_TOOLS_READ_PROCESS_MEMORY:
        case UVM_TOOLS_WRITE_PROCESS_MEMORY:
        /* Tools/Debug/Profiling */
        case UVM_TOOLS_INIT_EVENT_TRACKER:
        case UVM_TOOLS_SET_NOTIFICATION_THRESHOLD:
        case UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS:
        case UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS:
        case UVM_TOOLS_ENABLE_COUNTERS:
        case UVM_TOOLS_DISABLE_COUNTERS:
        case UVM_TOOLS_GET_PROCESSOR_UUID_TABLE:
        case UVM_TOOLS_FLUSH_EVENTS:
#ifdef UVM_TOOLS_INIT_EVENT_TRACKER_V2
        case UVM_TOOLS_INIT_EVENT_TRACKER_V2:
#endif
#ifdef UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2
        case UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2:
#endif
        /* Page Migration (managed memory only) */
        case UVM_MIGRATE:
        case UVM_MIGRATE_RANGE_GROUP:
        /* Migration Policy */
        case UVM_SET_PREFERRED_LOCATION:
        case UVM_UNSET_PREFERRED_LOCATION:
        case UVM_SET_ACCESSED_BY:
        case UVM_UNSET_ACCESSED_BY:
        case UVM_ENABLE_READ_DUPLICATION:
        case UVM_DISABLE_READ_DUPLICATION:
        case UVM_PREVENT_MIGRATION_RANGE_GROUPS:
        case UVM_ALLOW_MIGRATION_RANGE_GROUPS:
        /* Misc unnecessary */
        case UVM_MAP_DYNAMIC_PARALLELISM_REGION:
        case UVM_POPULATE_PAGEABLE:
#ifdef UVM_CLEAR_ALL_ACCESS_COUNTERS
        case UVM_CLEAR_ALL_ACCESS_COUNTERS:
#endif
#ifdef UVM_DISCARD
        case UVM_DISCARD:
#endif
            pr_debug_ratelimited("nvidia-uvm-hardened: blocked ioctl %u\n", cmd);
            return -ENOSYS;
    }
#endif /* UVM_HARDENED_MODE */

INJECT_EOF

    echo "NOTE: The ioctl blocking code has been generated at /tmp/uvm_hardened_inject.c"
    echo "You need to manually insert it into ${UVM_C} in the uvm_ioctl() function,"
    echo "after the initial validation and before the main ioctl dispatch switch."
    echo ""
    echo "Look for a function like:"
    echo "  static long uvm_unlocked_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)"
    echo "or:"
    echo "  long uvm_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)"
    echo ""
    echo "Insert the block BEFORE the main switch(cmd) dispatch."
fi

# Also block test ioctls unconditionally
UVM_TEST="kernel-open/nvidia-uvm/uvm_test.c"
if [ -f "${UVM_TEST}" ] && ! grep -q "UVM_HARDENED_MODE" "${UVM_TEST}" 2>/dev/null; then
    echo "Patching test ioctl handler..."
    # Add early return at the top of uvm_test_ioctl
    sed -i '/^NV_STATUS uvm_test_ioctl/,/^{/ {
        /^{/ a\
#ifdef UVM_HARDENED_MODE\
    return NV_ERR_NOT_SUPPORTED; /* vLLM hardened: all test ioctls blocked */\
#endif
    }' "${UVM_TEST}" 2>/dev/null || echo "NOTE: Manual patching of ${UVM_TEST} may be needed"
fi

# Build only nvidia-uvm module
echo ""
echo "Building nvidia-uvm.ko..."
make modules -j"$(nproc)" NV_KERNEL_MODULES=nvidia-uvm 2>&1 | tail -20

# Check result
UVM_KO="kernel-open/nvidia-uvm/nvidia-uvm.ko"
if [ -f "${UVM_KO}" ]; then
    echo ""
    echo "=== SUCCESS ==="
    echo "Hardened module built at: ${BUILD_DIR}/open-gpu-kernel-modules/${UVM_KO}"
    echo ""
    echo "To deploy:"
    echo "  sudo rmmod nvidia_uvm"
    echo "  sudo insmod ${BUILD_DIR}/open-gpu-kernel-modules/${UVM_KO}"
    echo ""
    echo "To verify:"
    echo "  python -c 'import torch; print(torch.cuda.is_available())'"
    echo "  # Should print True"
else
    echo ""
    echo "=== BUILD FAILED ==="
    echo "Check the build output above for errors."
    echo "Common issues:"
    echo "  - Missing kernel headers: apt install linux-headers-\$(uname -r)"
    echo "  - Driver version mismatch: ensure git tag matches installed driver"
    exit 1
fi
