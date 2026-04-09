# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant preset configuration.

Named presets encode compression settings directly in the kv_cache_dtype
string, following mgoin's request for "knobs to evaluate different
compression levels."

Preset format: tq_k{K}[f]v{V}[_qjl]
  K = key bits (2-8)
  f = FP8 key mode (K stored as fp8_e4m3, V as TQ; requires K=8)
  V = value bits (2-8)
  qjl = sign correction enabled (opt-in, not recommended for attention)

Recommended presets:
  tq_k8v8      = 8-bit MSE-only (best quality, 2x compression, default)
  tq_k8fv4     = FP8 keys + 4-bit TQ values (best quality/compression ratio)
  tq_k4v4      = 4-bit MSE-only (4x compression)

QJL (_qjl suffix) adds 1-bit residual sign correction. The paper uses
Algorithm 1 (MSE-only) for KV cache, not Algorithm 2 (QJL). Community
consensus: QJL adds variance that softmax amplifies, hurting quality.
Use MSE-only with norm correction (enabled by default) instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex as re

from vllm.v1.attention.ops.turboquant import (
    qjl_padded_dim,
    sign_bytes_padded,
)


@dataclass(frozen=True)
class TQPreset:
    """Parsed TurboQuant preset configuration."""

    name: str
    k_bits: int
    v_bits: int
    qjl: bool = False
    k_fp8: bool = False

    @property
    def k_byte_mode(self) -> bool:
        return self.k_fp8 or self.k_bits > 4

    @property
    def v_byte_mode(self) -> bool:
        return self.v_bits > 4

    @property
    def avg_bits_per_dim(self) -> float:
        return (self.k_bits + self.v_bits) / 2.0

    def cache_dim_per_head(self, head_dim: int, kv: str = "k") -> int:
        """Bytes per head for K or V cache.

        Standard mode:
          For byte mode (>4 bit): head_dim bytes (1 index per coord)
          For nibble mode (<=4 bit): head_dim // 2 bytes (2 indices per byte)
          Plus 4 bytes for L2 norm (float32).
          Plus QJL overhead if enabled.
        """
        if kv == "k" and self.k_fp8:
            # FP8 key: head_dim bytes (fp8_e4m3) + 4 bytes (float32 scale)
            return head_dim + 4

        bits = self.k_bits if kv == "k" else self.v_bits
        byte_mode = bits > 4

        idx_bytes = head_dim if byte_mode else head_dim // 2

        norm_bytes = 4  # float32 L2 norm

        if self.qjl:
            return (
                qjl_padded_dim(head_dim)
                if byte_mode
                else (idx_bytes + norm_bytes + sign_bytes_padded(head_dim) + 4)
            )

        return idx_bytes + norm_bytes

    def padded_cache_dim(self, head_dim: int) -> int:
        """Padded cache dimension per KV half.

        Uses max(k_dim, v_dim) so both K and V halves of the cache
        tensor have the same last dimension (required by standard
        paged KV cache shape with leading-2 dim).
        """
        k_dim = self.cache_dim_per_head(head_dim, "k")
        v_dim = self.cache_dim_per_head(head_dim, "v")
        raw = max(k_dim, v_dim)
        # Align to 16 bytes for memory access efficiency
        return (raw + 15) & ~15


# Regex for parsing preset strings
# 'f' after K bits = FP8 key mode
_TQ_PATTERN = re.compile(r"^tq_k(\d+)(f?)v(\d+)(_qjl)?$")


# Backward compat: old "turboquant" string maps to best-quality preset.
# Uses MSE-only (no QJL) — the paper's Algorithm 1 for KV cache.
_TQ_ALIASES = {
    "turboquant": "tq_k8v8",
    # Backward compat: accept hyphens from older command lines
    "tq-k8v8": "tq_k8v8",
    "tq-k8fv4": "tq_k8fv4",
    "tq-k4v4": "tq_k4v4",
    "tq-k8v8-qjl": "tq_k8v8_qjl",
    "tq-k4v4-qjl": "tq_k4v4_qjl",
}


def parse_tq_preset(kv_cache_dtype: str) -> TQPreset:
    """Parse a tq_* string into a TQPreset.

    Raises ValueError on invalid format.
    """
    kv_cache_dtype = _TQ_ALIASES.get(kv_cache_dtype, kv_cache_dtype)
    m = _TQ_PATTERN.match(kv_cache_dtype)
    if not m:
        raise ValueError(
            f"Invalid TurboQuant preset: '{kv_cache_dtype}'. "
            f"Expected format: tq_k{{K}}[f]v{{V}}[_qjl]. "
            f"Examples: tq_k8v8, tq_k8fv4, tq_k4v4_qjl"
        )
    k_bits = int(m.group(1))
    k_fp8 = m.group(2) == "f"
    v_bits = int(m.group(3))
    qjl = m.group(4) is not None

    if not (2 <= k_bits <= 8):
        raise ValueError(f"k_bits must be 2-8, got {k_bits}")
    if not (2 <= v_bits <= 8):
        raise ValueError(f"v_bits must be 2-8, got {v_bits}")
    if k_fp8 and k_bits != 8:
        raise ValueError(f"FP8 key mode requires k_bits=8, got {k_bits}")

    return TQPreset(
        name=kv_cache_dtype,
        k_bits=k_bits,
        v_bits=v_bits,
        qjl=qjl,
        k_fp8=k_fp8,
    )


def is_tq_preset(kv_cache_dtype: str) -> bool:
    """Return True if kv_cache_dtype is a tq_* preset or 'turboquant' alias."""
    kv_cache_dtype = _TQ_ALIASES.get(kv_cache_dtype, kv_cache_dtype)
    return bool(_TQ_PATTERN.match(kv_cache_dtype))


def get_boundary_skip_layers(num_layers: int, preset_name: str) -> list[str]:
    """Auto-compute boundary layers to skip for quality protection.

    First/last layers carry disproportionate semantic weight and are
    most sensitive to KV cache quantization error. Keeping them in bf16
    breaks the error accumulation chain for autoregressive generation.

    Heuristic (empirically validated on Qwen2.5-7B, Llama-3-8B):
      - 8-bit (avg > 6): near-lossless, no skip needed
      - Mixed (avg 4-6): protect first + last layer
      - 4-bit (avg <= 4): protect first 2 + last 2 layers
    """
    preset = parse_tq_preset(preset_name)
    avg_bits = preset.avg_bits_per_dim
    if avg_bits > 6:
        return []
    if avg_bits > 4:
        return [str(0), str(num_layers - 1)]
    # 4-bit and below: first 2 + last 2
    skip = {0, 1, num_layers - 2, num_layers - 1}
    return sorted([str(x) for x in skip if 0 <= x < num_layers], key=int)
