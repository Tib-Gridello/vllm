# vLLM nvidia_uvm Security Hardening

## Threat Model

The `nvidia_uvm` kernel module (`/dev/nvidia-uvm`) exposes ~45+ ioctls to userspace,
including `UVM_TOOLS_READ_PROCESS_MEMORY` and `UVM_TOOLS_WRITE_PROCESS_MEMORY` which
allow arbitrary process memory access. A single shared device node provides access to
memory management across ALL GPUs on the system.

See: [Synacktiv - Deploying an LLM Server with Least Privileges](https://www.synacktiv.com/publications/grand-saut-dans-le-deploiement-sur-site-dun-serveur-llm-a-moindres-privileges)

## Seccomp BPF Profile

`uvm_seccomp_profile.json` blocks 26 dangerous nvidia_uvm ioctls while allowing the
~15 required for basic CUDA operation (cuMemAlloc, kernel launch, multi-GPU P2P).

### Blocked ioctl categories

| Category | Ioctls Blocked | Risk Level |
|----------|---------------|------------|
| Tools/Debug | READ/WRITE_PROCESS_MEMORY, event trackers, counters, UUID tables | CRITICAL |
| Page Migration | MIGRATE, MIGRATE_RANGE_GROUP, PREVENT/ALLOW_MIGRATION | MEDIUM |
| Managed Memory | SET/UNSET_PREFERRED_LOCATION, ACCESSED_BY, READ_DUPLICATION | MEDIUM |
| Misc | DYNAMIC_PARALLELISM, POPULATE_PAGEABLE, DISCARD, ACCESS_COUNTERS | LOW |

### Allowed ioctls (required for CUDA)

- UVM_INITIALIZE / UVM_DEINITIALIZE (0x30000001 / 0x30000002)
- UVM_MM_INITIALIZE (75)
- UVM_REGISTER_GPU / UVM_UNREGISTER_GPU (37/38)
- UVM_REGISTER_GPU_VASPACE / UVM_UNREGISTER_GPU_VASPACE (25/26)
- UVM_REGISTER_CHANNEL / UVM_UNREGISTER_CHANNEL (27/28)
- UVM_CREATE_EXTERNAL_RANGE (73)
- UVM_MAP_EXTERNAL_ALLOCATION (33)
- UVM_UNMAP_EXTERNAL (66)
- UVM_FREE (34)
- UVM_ENABLE/DISABLE_PEER_ACCESS (29/30)
- UVM_CREATE/DESTROY_RANGE_GROUP (23/24)
- UVM_SET_RANGE_GROUP (31)
- UVM_ALLOC_SEMAPHORE_POOL (68)
- UVM_PAGEABLE_MEM_ACCESS (39) — called by some CUDA versions during init
- UVM_VALIDATE_VA_RANGE (72)

### Usage

#### Docker
```bash
docker run --gpus all \
  --security-opt seccomp=security/uvm_seccomp_profile.json \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B
```

#### Podman
```bash
podman run --device nvidia.com/gpu=all \
  --security-opt seccomp=security/uvm_seccomp_profile.json \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B
```

#### Kubernetes (via seccomp profile)
```yaml
securityContext:
  seccompProfile:
    type: Localhost
    localhostProfile: profiles/vllm-uvm-hardened.json
```

### Caveat: ioctl number collisions

Seccomp BPF filters on the ioctl `cmd` argument globally — not just for `/dev/nvidia-uvm`.
The blocked ioctl numbers (40-80 range) could theoretically collide with ioctls on other
device drivers. In practice, this is unlikely in a GPU inference container that only
interacts with NVIDIA devices and standard Linux fds.

If you encounter issues with other device drivers, use `SECCOMP_RET_USER_NOTIF` with a
supervisor process that inspects the fd target before blocking.

## Custom Hardened nvidia_uvm.ko

For deeper protection (blocking at the kernel module level, not just syscall level),
see `security/nvidia-uvm-hardened/` for a patch against NVIDIA open-source GPU kernel
modules that stubs out dangerous ioctls at the source.
