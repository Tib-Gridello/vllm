# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant preset configuration.

Named presets encode compression settings directly in the kv_cache_dtype
string, following mgoin's request for "knobs to evaluate different
compression levels."

Preset format: tq-k{K}v{V}[-qjl][-o]
  K = key bits (2-8)
  V = value bits (2-8)
  qjl = sign correction enabled (opt-in, not recommended for attention)
  o = outlier channel mode (mixed-precision: outlier@K bits, regular@2 bits)

Recommended presets:
  tq-k8v8      = 8-bit MSE-only (best quality, 2x compression, default)
  tq-k4v4      = 4-bit MSE-only (4x compression)
  tq-k4v2o     = outlier mode: 4-bit outlier channels + 2-bit regular (paper)

QJL (-qjl suffix) adds 1-bit residual sign correction. The paper uses
Algorithm 1 (MSE-only) for KV cache, not Algorithm 2 (QJL). Community
consensus: QJL adds variance that softmax amplifies, hurting quality.
Use MSE-only with norm correction (enabled by default) instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
    outlier_mode: bool = False
    outlier_ratio: float = 0.25

    @property
    def k_byte_mode(self) -> bool:
        return self.k_bits > 4

    @property
    def v_byte_mode(self) -> bool:
        return self.v_bits > 4

    @property
    def avg_bits_per_dim(self) -> float:
        if self.outlier_mode:
            # Outlier channels at k_bits, regular at 2 bits
            return self.outlier_ratio * self.k_bits + (1 - self.outlier_ratio) * 2
        return (self.k_bits + self.v_bits) / 2.0

    def cache_dim_per_head(self, head_dim: int, kv: str = "k") -> int:
        """Bytes per head for K or V cache.

        Standard mode:
          For byte mode (>4 bit): head_dim bytes (1 index per coord)
          For nibble mode (<=4 bit): head_dim // 2 bytes (2 indices per byte)
          Plus 4 bytes for L2 norm (float32).
          Plus QJL overhead if enabled.

        Outlier mode:
          Outlier indices (nibble/2-bit packed) + Regular indices (2-bit packed)
          Plus 8 bytes for dual L2 norms (2 × float32).
          Both K and V use the same outlier layout.
        """
        if self.outlier_mode:
            outlier_dim = int(head_dim * self.outlier_ratio)
            regular_dim = head_dim - outlier_dim
            # k_bits = outlier bits, v_bits = regular bits (preset convention)
            out_bits = self.k_bits
            reg_bits = self.v_bits
            if out_bits <= 2:
                out_idx_bytes = (outlier_dim + 3) // 4
            elif out_bits <= 4:
                out_idx_bytes = outlier_dim // 2
            else:
                out_idx_bytes = outlier_dim
            if reg_bits <= 2:
                reg_idx_bytes = (regular_dim + 3) // 4
            elif reg_bits <= 4:
                reg_idx_bytes = regular_dim // 2
            else:
                reg_idx_bytes = regular_dim
            norm_bytes = 8  # 2 × float32 (outlier_norm + regular_norm)
            return out_idx_bytes + reg_idx_bytes + norm_bytes

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
_TQ_PATTERN = re.compile(r"^tq-k(\d+)v(\d+)(-qjl)?(o|-o)?$")


# Backward compat: old "turboquant" string maps to best-quality preset.
# Uses MSE-only (no QJL) — the paper's Algorithm 1 for KV cache.
_TQ_ALIASES = {
    "turboquant": "tq-k8v8",
}


def parse_tq_preset(kv_cache_dtype: str) -> TQPreset:
    """Parse a tq-* string into a TQPreset.

    Raises ValueError on invalid format.
    """
    kv_cache_dtype = _TQ_ALIASES.get(kv_cache_dtype, kv_cache_dtype)
    m = _TQ_PATTERN.match(kv_cache_dtype)
    if not m:
        raise ValueError(
            f"Invalid TurboQuant preset: '{kv_cache_dtype}'. "
            f"Expected format: tq-k{{K}}v{{V}}[-qjl][-o]. "
            f"Examples: tq-k8v8, tq-k4v4-qjl, tq-k4v2o"
        )
    k_bits = int(m.group(1))
    v_bits = int(m.group(2))
    qjl = m.group(3) is not None
    outlier = m.group(4) is not None

    if not (2 <= k_bits <= 8):
        raise ValueError(f"k_bits must be 2-8, got {k_bits}")
    if not (2 <= v_bits <= 8):
        raise ValueError(f"v_bits must be 2-8, got {v_bits}")

    return TQPreset(
        name=kv_cache_dtype,
        k_bits=k_bits,
        v_bits=v_bits,
        qjl=qjl,
        outlier_mode=outlier,
    )


def is_tq_preset(kv_cache_dtype: str) -> bool:
    """Return True if kv_cache_dtype is a tq-* preset or 'turboquant' alias."""
    kv_cache_dtype = _TQ_ALIASES.get(kv_cache_dtype, kv_cache_dtype)
    return bool(_TQ_PATTERN.match(kv_cache_dtype))
