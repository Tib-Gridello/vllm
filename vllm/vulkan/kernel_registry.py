# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Vulkan Kernel Registry — manages PTX modules, binary cache, and dispatch.

Maps vLLM operation names to Vulkan-launched PTX kernels. Handles:
- Loading PTX from files or embedded strings
- Binary cache for fast reload (skip JIT compilation)
- Dtype dispatch (f32/f16/bf16)
- Compute capability matching (sm_80, sm_89, sm_90, etc.)
"""

import hashlib
import os
import struct
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# PTX kernel directory (relative to this file)
_PTX_DIR = Path(__file__).parent.parent.parent / "csrc" / "vulkan" / "ptx_kernels"
_CACHE_DIR = Path(
    os.environ.get("VLLM_VULKAN_CACHE_DIR",
                   os.path.expanduser("~/.cache/vllm/vulkan_ptx_cache"))
)


class KernelSpec:
    """Specification for a single PTX kernel function."""

    def __init__(self, module_name: str, function_name: str,
                 param_types: list[str]):
        """
        Args:
            module_name: Name of the PTX module (e.g. "activation_kernels")
            function_name: Name of the extern "C" function in PTX
            param_types: List of parameter type strings for struct.pack
                         'Q' = uint64 (pointer/device address)
                         'i' = int32
                         'I' = uint32
                         'f' = float32
                         'q' = int64
        """
        self.module_name = module_name
        self.function_name = function_name
        self.param_types = param_types
        self.pack_fmt = "<" + "".join(param_types)

    def pack_params(self, *args) -> list[bytes]:
        """Pack kernel parameters as a list of bytes objects."""
        result = []
        for typ, val in zip(self.param_types, args):
            result.append(struct.pack("<" + typ, val))
        return result


# Registry of all available Vulkan kernel functions.
# Maps: (op_name, dtype) -> KernelSpec
_KERNEL_REGISTRY: dict[tuple[str, str], KernelSpec] = {}


def _register(op: str, dtype: str, module: str, func: str,
              params: list[str]):
    _KERNEL_REGISTRY[(op, dtype)] = KernelSpec(module, func, params)


# ============================================================================
# Register all standalone kernels
# ============================================================================

# --- Activation kernels ---
# silu_and_mul: (out*, input*, d) -> ()
for _dtype, _suffix in [("f32", "f32"), ("f16", "f16"), ("bf16", "bf16")]:
    _register("silu_and_mul", _dtype, "activation_kernels",
              f"silu_and_mul_{_suffix}", ["Q", "Q", "i"])
    _register("gelu_and_mul", _dtype, "activation_kernels",
              f"gelu_and_mul_{_suffix}", ["Q", "Q", "i"])
    _register("gelu_tanh_and_mul", _dtype, "activation_kernels",
              f"gelu_tanh_and_mul_{_suffix}", ["Q", "Q", "i"])

_register("silu_activation", "f32", "activation_kernels",
          "silu_activation_f32", ["Q", "Q", "i"])
_register("gelu_activation", "f32", "activation_kernels",
          "gelu_activation_f32", ["Q", "Q", "i"])
_register("relu_activation", "f32", "activation_kernels",
          "relu_activation_f32", ["Q", "Q", "i"])

# --- LayerNorm kernels ---
# rms_norm: (out*, input*, weight*, epsilon, hidden_size) -> ()
for _dtype, _suffix in [("f32", "f32"), ("f16", "f16"), ("bf16", "bf16")]:
    _register("rms_norm", _dtype, "layernorm_kernels",
              f"rms_norm_{_suffix}", ["Q", "Q", "Q", "f", "i"])

_register("fused_add_rms_norm", "f32", "layernorm_kernels",
          "fused_add_rms_norm_f32", ["Q", "Q", "Q", "f", "i"])

# --- RoPE kernels ---
# rotary_embedding: (positions*, query*, key*, cos_sin_cache*,
#                    rot_dim, query_stride, key_stride, head_stride,
#                    num_heads, num_kv_heads, head_size) -> ()
for _dtype, _suffix in [("f32", "f32"), ("f16", "f16"), ("bf16", "bf16")]:
    _register("rotary_embedding_neox", _dtype, "pos_encoding_kernels",
              f"rotary_embedding_neox_{_suffix}",
              ["Q", "Q", "Q", "Q", "i", "q", "q", "q", "i", "i", "i"])

for _dtype, _suffix in [("f16", "f16"), ("bf16", "bf16")]:
    _register("rotary_embedding_gptj", _dtype, "pos_encoding_kernels",
              f"rotary_embedding_gptj_{_suffix}",
              ["Q", "Q", "Q", "Q", "i", "q", "q", "q", "i", "i", "i"])

# --- Cache kernels ---
# copy_blocks: (key_cache_ptrs*, value_cache_ptrs*, block_mapping*,
#               numel_per_block) -> ()
for _dtype, _suffix in [("f16", "f16"), ("bf16", "bf16")]:
    _register("copy_blocks", _dtype, "cache_kernels",
              f"copy_blocks_{_suffix}", ["Q", "Q", "Q", "i"])
    _register("reshape_and_cache", _dtype, "cache_kernels",
              f"reshape_and_cache_{_suffix}",
              ["Q", "Q", "Q", "Q", "Q", "i", "i", "i", "i", "i", "i"])

# --- Sampler kernels ---
_register("apply_repetition_penalties", "f32", "sampler_kernels",
          "apply_repetition_penalties_f32",
          ["Q", "Q", "Q", "Q", "i", "i", "i"])
_register("apply_repetition_penalties", "f16", "sampler_kernels",
          "apply_repetition_penalties_f16",
          ["Q", "Q", "Q", "Q", "i", "i", "i"])
_register("temperature_scale", "f32", "sampler_kernels",
          "temperature_scale_f32", ["Q", "Q", "i", "i"])
_register("softmax", "f32", "sampler_kernels",
          "softmax_f32", ["Q", "Q", "i", "i"])
_register("multinomial_sample", "f32", "sampler_kernels",
          "multinomial_sample_f32", ["Q", "Q", "Q", "i", "i"])

# --- Paged Attention kernels ---
# Params: (out*, q*, k_cache*, v_cache*, num_kv_heads, scale,
#          block_tables*, seq_lens*, max_num_blocks_per_seq,
#          q_stride, kv_block_stride, kv_head_stride)
_register("paged_attention_v1_h128_b16", "f32", "paged_attention_kernels",
          "paged_attention_v1_h128_b16_t128",
          ["Q", "Q", "Q", "Q", "i", "f", "Q", "Q", "i", "i", "i", "i"])
_register("paged_attention_v1_h64_b16", "f32", "paged_attention_kernels",
          "paged_attention_v1_h64_b16_t128",
          ["Q", "Q", "Q", "Q", "i", "f", "Q", "Q", "i", "i", "i", "i"])
_register("paged_attention_v1_h128_b32", "f32", "paged_attention_kernels",
          "paged_attention_v1_h128_b32_t128",
          ["Q", "Q", "Q", "Q", "i", "f", "Q", "Q", "i", "i", "i", "i"])

# f16 paged attention
_PA_PARAMS = ["Q", "Q", "Q", "Q", "i", "f", "Q", "Q", "i", "i", "i", "i"]
_register("paged_attention_v1_h128_b16", "f16", "paged_attention_kernels",
          "paged_attention_v1_f16_h128_b16_t128", _PA_PARAMS)
_register("paged_attention_v1_h64_b16", "f16", "paged_attention_kernels",
          "paged_attention_v1_f16_h64_b16_t128", _PA_PARAMS)

# bf16 paged attention
_register("paged_attention_v1_h128_b16", "bf16", "paged_attention_kernels",
          "paged_attention_v1_bf16_h128_b16_t128", _PA_PARAMS)
_register("paged_attention_v1_h64_b16", "bf16", "paged_attention_kernels",
          "paged_attention_v1_bf16_h64_b16_t128", _PA_PARAMS)

# --- GEMM kernels (fallback when cuBLAS unavailable) ---
# gemm_nn: (C*, A*, B*, M, N, K, alpha, beta)
_register("gemm_nn", "f32", "gemm_kernels", "gemm_nn_f32",
          ["Q", "Q", "Q", "i", "i", "i", "f", "f"])
_register("gemm_nn", "f16", "gemm_kernels", "gemm_nn_f16",
          ["Q", "Q", "Q", "i", "i", "i", "f", "f"])

# linear: (output*, input*, weight*, bias*, M, N, K)
_register("linear", "f16", "gemm_kernels", "linear_f16",
          ["Q", "Q", "Q", "Q", "i", "i", "i"])
_register("linear", "bf16", "gemm_kernels", "linear_bf16",
          ["Q", "Q", "Q", "Q", "i", "i", "i"])


def get_kernel_spec(op_name: str, dtype: str) -> KernelSpec | None:
    """Look up a kernel spec by operation name and dtype."""
    return _KERNEL_REGISTRY.get((op_name, dtype))


def list_registered_kernels() -> list[tuple[str, str]]:
    """List all registered (op_name, dtype) pairs."""
    return sorted(_KERNEL_REGISTRY.keys())


class VulkanKernelManager:
    """
    Manages loaded PTX modules and provides kernel dispatch for a device.

    Handles PTX loading, binary caching, and kernel function lookup.
    Uses a two-level cache:
      1. In-memory: loaded VkCudaModuleNV + VkCudaFunctionNV handles
      2. On-disk: compiled binary cache files (skip JIT on reload)
    """

    def __init__(self, device: Any, sm_arch: int):
        """
        Args:
            device: VulkanDevice instance
            sm_arch: GPU SM architecture (e.g. 80 for Ampere, 90 for Hopper)
        """
        from vllm.vulkan import KernelFunction, KernelModule

        self.device = device
        self.sm_arch = sm_arch
        self._modules: dict[str, KernelModule] = {}
        self._functions: dict[str, KernelFunction] = {}

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_function(self, op_name: str, dtype: str) -> Any:
        """Get a ready-to-launch KernelFunction for the given op and dtype."""
        from vllm.vulkan import KernelFunction, KernelModule

        spec = get_kernel_spec(op_name, dtype)
        if spec is None:
            raise ValueError(f"No Vulkan kernel registered for "
                             f"op='{op_name}', dtype='{dtype}'")

        func_key = f"{spec.module_name}::{spec.function_name}"

        if func_key in self._functions:
            return self._functions[func_key], spec

        # Load the module if not already loaded
        if spec.module_name not in self._modules:
            self._modules[spec.module_name] = self._load_module(
                spec.module_name)

        # Create the function
        func = KernelFunction(self.device, self._modules[spec.module_name],
                              spec.function_name)
        self._functions[func_key] = func
        logger.debug("Loaded Vulkan kernel: %s", func_key)
        return func, spec

    def _load_module(self, module_name: str) -> Any:
        """Load a PTX module, using binary cache if available."""
        from vllm.vulkan import KernelModule

        cache_path = _CACHE_DIR / f"{module_name}.sm{self.sm_arch}.cache"

        # Try loading from binary cache first (fast path)
        if cache_path.exists():
            try:
                cache_data = cache_path.read_bytes()
                module = KernelModule(self.device, cache_data)
                logger.info("Loaded %s from binary cache", module_name)
                return module
            except Exception:
                logger.warning("Binary cache invalid for %s, recompiling",
                               module_name)
                cache_path.unlink(missing_ok=True)

        # Load from PTX source (slow path — triggers JIT in Vulkan ICD)
        ptx_path = (_PTX_DIR / "generated" /
                    f"{module_name}.sm{self.sm_arch}.ptx")
        if not ptx_path.exists():
            # Try without SM suffix
            ptx_path = _PTX_DIR / "generated" / f"{module_name}.ptx"

        if not ptx_path.exists():
            raise FileNotFoundError(
                f"PTX file not found: {ptx_path}. "
                f"Run: cd csrc/vulkan/ptx_kernels && "
                f"./extract_ptx.sh {self.sm_arch}")

        ptx_source = ptx_path.read_text()
        logger.info("Loading %s from PTX (%d bytes, JIT compiling...)",
                     module_name, len(ptx_source))

        module = KernelModule(self.device, ptx_source)

        # Save binary cache for next time
        try:
            cache_data = module.get_binary_cache()
            cache_path.write_bytes(cache_data)
            logger.info("Saved binary cache for %s (%d bytes)",
                         module_name, len(cache_data))
        except Exception as e:
            logger.warning("Failed to save binary cache: %s", e)

        return module
