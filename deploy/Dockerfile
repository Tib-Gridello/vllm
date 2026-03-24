# vLLM Hardened — nvidia_uvm attack surface elimination
#
# Reproduces the Synacktiv least-privilege LLM deployment:
# https://www.synacktiv.com/publications/grand-saut-dans-le-deploiement-sur-site-dun-serveur-llm-a-moindres-privileges
#
# Security features:
#   - seccomp profile blocks 26 dangerous nvidia_uvm ioctls
#   - Non-root runtime user
#   - Dropped capabilities
#   - Vulkan backend (compile + PTX extraction) for future full nvidia_uvm elimination
#   - SSH for remote management

ARG VLLM_TAG=v0.17.0
FROM vllm/vllm-openai:${VLLM_TAG}

# ============================================================
# 1. System packages: SSH + Vulkan runtime + build tools
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server \
        libvulkan1 \
        vulkan-tools \
        strace \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && ssh-keygen -A

# ============================================================
# 2. Copy our security hardening
# ============================================================
COPY vllm/security/uvm_seccomp_profile.json /etc/vllm/seccomp.json
COPY vllm/security/README.md /etc/vllm/security-README.md

# ============================================================
# 3. Copy Vulkan backend code (Python modules)
# ============================================================
COPY vllm/vllm/vulkan/ /workspace/vllm-uvm/vulkan/

# ============================================================
# 4. Copy standalone CUDA kernels for PTX extraction
# ============================================================
COPY vllm/csrc/vulkan/ptx_kernels/ /workspace/vllm-uvm/ptx_kernels/

# ============================================================
# 5. Copy test scripts
# ============================================================
COPY deploy/test_hardened.sh /workspace/test_hardened.sh
RUN chmod +x /workspace/test_hardened.sh

# ============================================================
# 6. Entrypoint: SSH + configurable mode
# ============================================================
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Environment
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all

EXPOSE 22 8000

ENTRYPOINT ["/entrypoint.sh"]
