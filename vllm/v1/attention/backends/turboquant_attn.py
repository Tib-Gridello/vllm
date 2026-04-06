# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone TurboQuant KV Cache Compression Backend.

Implements Algorithm 1 (MSE-only) from "TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874).

Architecture: fused attention — the Triton unified attention kernel reads
compressed indices inline, looks up centroids, applies norm-corrected
L2 norms, and computes attention in a single pass. No staging buffer.

Key design choices:
- Hadamard rotation (3pp MMLU improvement over random QR)
- Norm correction (compensates centroid vector norm != 1)
- Exact Lloyd-Max codebook (scipy numerical integration, not Monte Carlo)
- MSE-only by default (paper's Algorithm 1 for KV cache; QJL opt-in)
- Fused decode (no decompression to bf16 staging)
- Named presets via --kv-cache-dtype

Usage:
  vllm serve <model> --kv-cache-dtype tq-k8v8      # 8-bit, 2x compression
  vllm serve <model> --kv-cache-dtype tq-k4v4      # 4-bit, 4x compression
  vllm serve <model> --kv-cache-dtype tq-k8v8-qjl  # with sign correction (opt-in)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.turboquant_config import (
    is_tq_preset,
    parse_tq_preset,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Attention metadata for TurboQuant backend."""

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


# ---------------------------------------------------------------------------
# Metadata Builder
# ---------------------------------------------------------------------------


class TurboQuantMetadataBuilder(
    AttentionMetadataBuilder[TurboQuantMetadata],
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.device = device
        self.block_size = kv_cache_spec.block_size

    def reorder_batch(self, input_batch, scheduler_output) -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TurboQuantMetadata:
        return TurboQuantMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> TurboQuantMetadata:
        return self.build(0, common_attn_metadata, fast_build=True)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TurboQuantAttentionBackend(AttentionBackend):
    """Standalone TurboQuant KV cache compression backend."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type[TurboQuantAttentionImpl]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TurboQuantMetadataBuilder]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "tq-k8v8-qjl",
    ) -> tuple[int, ...]:
        """KV cache shape: (num_blocks, 2, block_size, num_kv_heads, padded_dim).

        Standard leading-2 for K/V compatibility with block manager,
        KV transfer, prefix caching.
        """
        preset = parse_tq_preset(cache_dtype_str)
        padded = preset.padded_cache_dim(head_size)
        return (num_blocks, 2, block_size, num_kv_heads, padded)

    @classmethod
    def supports_kv_cache_dtype(
        cls,
        kv_cache_dtype: CacheDType | None,
    ) -> bool:
        if kv_cache_dtype is None:
            return False
        return is_tq_preset(kv_cache_dtype)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Hadamard requires power-of-2 head_dim
        return head_size >= 32 and (head_size & (head_size - 1)) == 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class TurboQuantAttentionImpl(AttentionImpl[TurboQuantMetadata]):
    """TurboQuant attention: fused decode with inline dequant.

    Encode (do_kv_cache_update):
      K/V → normalize → Hadamard rotate → quantize → pack → scatter

    Forward (fused attention):
      Rotate Q → unified_attention reads compressed cache inline
      (centroid lookup + norm + QJL correction inside attention loop)
      → inverse-rotate output
    """

    # Per-instance cache views (point into each layer's KV cache)
    _k_norms: torch.Tensor | None = None
    _v_norms: torch.Tensor | None = None
    _k_signs: torch.Tensor | None = None
    _v_signs: torch.Tensor | None = None
    _k_res_scales: torch.Tensor | None = None
    _v_res_scales: torch.Tensor | None = None
    _norms_dirty: bool = False

    # Outlier mode state
    _outlier_mode: bool = False
    _k_out_norms: torch.Tensor | None = None
    _k_reg_norms: torch.Tensor | None = None
    _v_out_norms: torch.Tensor | None = None
    _v_reg_norms: torch.Tensor | None = None
    _staging_k: torch.Tensor | None = None
    _staging_v: torch.Tensor | None = None
    _out_rotated_buf: torch.Tensor | None = None
    _reg_rotated_buf: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.alibi_slopes = alibi_slopes
        self.use_alibi_sqrt = use_alibi_sqrt
        self.attn_type = attn_type

        if sliding_window is not None:
            self.sliding_window = (sliding_window, sliding_window)
        else:
            self.sliding_window = (-1, -1)

        # Parse preset
        self._preset = parse_tq_preset(kv_cache_dtype)
        self._outlier_mode = self._preset.outlier_mode

        if self._outlier_mode:
            # Outlier mode: dequant-first architecture.
            # Compressed cache → Triton dequant → bf16 staging → standard attention.
            # Configs are created with default mask initially; calibration
            # replaces the mask on the first forward pass using actual K/V
            # channel variances.
            from vllm.v1.attention.ops.turboquant import OutlierChannelConfig

            self._kv_quant_mode = KVQuantMode.NONE  # staging is bf16
            self._needs_calibration = True

            # Default configs (will be replaced by calibrated ones)
            self._k_outlier_config = OutlierChannelConfig(
                head_dim=head_size,
                outlier_ratio=self._preset.outlier_ratio,
                outlier_bits=self._preset.k_bits,
                regular_bits=self._preset.v_bits,
                seed=42,
                device="cpu",
            )
            self._v_outlier_config = OutlierChannelConfig(
                head_dim=head_size,
                outlier_ratio=self._preset.outlier_ratio,
                outlier_bits=self._preset.k_bits,
                regular_bits=self._preset.v_bits,
                seed=43,  # different rotation for V
                device="cpu",
            )

            logger.info(
                "TurboQuant backend: preset=%s, outlier_bits=%d, "
                "regular_bits=%d, outlier_ratio=%.2f, avg_bits=%.1f, "
                "head_dim=%d, padded_dim=%d, mode=dequant-first"
                " (calibration pending)",
                self._preset.name,
                self._preset.k_bits,
                self._preset.v_bits,
                self._preset.outlier_ratio,
                self._preset.avg_bits_per_dim,
                head_size,
                self._preset.padded_cache_dim(head_size),
            )
        else:
            # Standard mode: fused attention with inline dequant.
            if self._preset.k_byte_mode != self._preset.v_byte_mode:
                raise NotImplementedError(
                    "Mixed byte/nibble K/V not yet supported. "
                    f"k_bits={self._preset.k_bits}, "
                    f"v_bits={self._preset.v_bits}. "
                    "Both must be >4 (byte) or both <=4 (nibble)."
                )

            if self._preset.k_byte_mode:
                self._kv_quant_mode = KVQuantMode.TURBOQUANT_BYTE
            else:
                self._kv_quant_mode = KVQuantMode.TURBOQUANT

            from vllm.v1.attention.ops.turboquant import TurboQuantCodebook

            self._k_codebook = TurboQuantCodebook(
                n_bits=self._preset.k_bits,
                head_dim=head_size,
                seed=42,
                device="cpu",
                qjl=self._preset.qjl,
            )
            self._v_codebook = TurboQuantCodebook(
                n_bits=self._preset.v_bits,
                head_dim=head_size,
                seed=43,
                device="cpu",
                qjl=self._preset.qjl,
            )

            logger.info(
                "TurboQuant backend: preset=%s, k_bits=%d, v_bits=%d, "
                "qjl=%s, head_dim=%d, padded_dim=%d, fused=True",
                self._preset.name,
                self._preset.k_bits,
                self._preset.v_bits,
                self._preset.qjl,
                head_size,
                self._preset.padded_cache_dim(head_size),
            )

    # ----- Cache view management -----

    def _ensure_cache_views(self, kv_cache: torch.Tensor) -> None:
        """Extract norm/sign/res_scale views from cache padding.

        Cache shape: (num_blocks, 2, block_size, nkv, padded_dim)
        Standard layout: [indices | L2 norm (4B) | sign bits | res_scale (4B)]
        Outlier layout: [out_indices | reg_indices | out_norm (4B) | reg_norm (4B)]
        """
        if self._outlier_mode:
            if self._k_out_norms is not None:
                return
            self._ensure_outlier_cache_views(kv_cache)
            return

        if self._k_norms is not None:
            return

        num_blocks, _, block_size, nkv, padded_dim = kv_cache.shape
        raw = kv_cache.untyped_storage()

        # Float32 view for norms
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        kv_half_bytes = block_size * nkv * padded_dim
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim // 4
        head_f32 = padded_dim // 4

        # Process K and V with their respective codebooks
        for kv_idx, cb, attr_prefix in [
            (0, self._k_codebook, "_k"),
            (1, self._v_codebook, "_v"),
        ]:
            hd = cb.head_dim
            idx_bytes = hd if cb.byte_mode else hd // 2
            norm_off_f32 = idx_bytes // 4
            kv_offset_f32 = kv_idx * kv_half_bytes // 4

            norms = torch.as_strided(
                base_f32,
                size=(num_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=kv_offset_f32 + norm_off_f32,
            )
            norms.fill_(0.0)
            setattr(self, f"{attr_prefix}_norms", norms)

            if cb.qjl:
                from vllm.v1.attention.ops.turboquant import sign_bytes_padded

                sbytes = sign_bytes_padded(hd)

                # Float32 view for res_scale
                res_off_f32 = (idx_bytes + 4 + sbytes) // 4
                res_scales = torch.as_strided(
                    base_f32,
                    size=(num_blocks, block_size, nkv),
                    stride=(full_block_f32, slot_f32, head_f32),
                    storage_offset=kv_offset_f32 + res_off_f32,
                )
                res_scales.fill_(0.0)
                setattr(self, f"{attr_prefix}_res_scales", res_scales)

                # Uint8 view for sign bits (still needed for encode path)
                base_u8 = torch.tensor(
                    [], dtype=torch.uint8, device=kv_cache.device
                ).set_(raw)
                kv_off_u8 = kv_idx * kv_half_bytes
                sign_byte_off = idx_bytes + 4  # after indices + norm

                signs = torch.as_strided(
                    base_u8,
                    size=(num_blocks, block_size, nkv, sbytes),
                    stride=(2 * kv_half_bytes, nkv * padded_dim, padded_dim, 1),
                    storage_offset=kv_off_u8 + sign_byte_off,
                )
                signs.fill_(0)
                setattr(self, f"{attr_prefix}_signs", signs)

    def _ensure_outlier_cache_views(self, kv_cache: torch.Tensor) -> None:
        """Extract dual norm views for outlier mode.

        Outlier cache layout per head:
          [outlier_packed | regular_packed | outlier_norm(4B) | regular_norm(4B)]
        """
        num_blocks, _, block_size, nkv, padded_dim = kv_cache.shape
        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor(
            [], dtype=torch.float32, device=kv_cache.device
        ).set_(raw)

        kv_half_bytes = block_size * nkv * padded_dim
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim // 4
        head_f32 = padded_dim // 4

        # Compute index bytes for outlier layout
        config = self._k_outlier_config  # same layout for K and V
        out_bytes = config.outlier_dim // 2  # nibble packed
        reg_bytes = (config.regular_dim + 3) // 4  # 2-bit packed
        idx_total = out_bytes + reg_bytes

        # Outlier norm at byte offset idx_total, Regular norm at idx_total + 4
        out_norm_off_f32 = idx_total // 4
        reg_norm_off_f32 = (idx_total + 4) // 4

        for kv_idx, attr_prefix in [(0, "_k"), (1, "_v")]:
            kv_offset_f32 = kv_idx * kv_half_bytes // 4

            out_norms = torch.as_strided(
                base_f32,
                size=(num_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=kv_offset_f32 + out_norm_off_f32,
            )
            out_norms.fill_(0.0)
            setattr(self, f"{attr_prefix}_out_norms", out_norms)

            reg_norms = torch.as_strided(
                base_f32,
                size=(num_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=kv_offset_f32 + reg_norm_off_f32,
            )
            reg_norms.fill_(0.0)
            setattr(self, f"{attr_prefix}_reg_norms", reg_norms)

    def _ensure_staging_buffers(
        self, kv_cache: torch.Tensor, n_staging_blocks: int,
    ) -> None:
        """Allocate/resize staging buffers for outlier dequant-first mode.

        Only allocates for `n_staging_blocks` (= num_seqs * max_blocks_per_seq),
        NOT the full cache capacity. This keeps memory usage proportional to the
        active batch, preserving the compression memory savings.
        """
        _, _, block_size, nkv, _ = kv_cache.shape
        head_dim = self.head_size
        device = kv_cache.device

        # Only reallocate if current buffers are too small
        if self._staging_k is not None and self._staging_k.shape[0] >= n_staging_blocks:
            return

        # Staging: bf16 paged cache for decompressed data
        self._staging_k = torch.zeros(
            (n_staging_blocks, block_size, nkv, head_dim),
            dtype=torch.bfloat16, device=device,
        )
        self._staging_v = torch.zeros(
            (n_staging_blocks, block_size, nkv, head_dim),
            dtype=torch.bfloat16, device=device,
        )

        # Intermediate buffers for rotated centroid values
        outlier_dim = self._k_outlier_config.outlier_dim
        regular_dim = self._k_outlier_config.regular_dim
        self._out_rotated_buf = torch.zeros(
            (n_staging_blocks, block_size, nkv, outlier_dim),
            dtype=torch.float32, device=device,
        )
        self._reg_rotated_buf = torch.zeros(
            (n_staging_blocks, block_size, nkv, regular_dim),
            dtype=torch.float32, device=device,
        )

    def _reset_cache_views(self) -> None:
        self._k_norms = None
        self._v_norms = None
        self._k_signs = None
        self._v_signs = None
        self._k_res_scales = None
        self._v_res_scales = None
        self._k_out_norms = None
        self._k_reg_norms = None
        self._v_out_norms = None
        self._v_reg_norms = None

    # ----- Forward (fused attention) -----

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(query)

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        if kv_cache.numel() == 0:
            self._norms_dirty = True
            return output.fill_(0)

        self._ensure_cache_views(kv_cache)

        if self._outlier_mode:
            return self._forward_outlier(
                query, kv_cache, attn_metadata, output, num_actual_tokens
            )

        from vllm.v1.attention.ops.turboquant import (
            inverse_rotate_output,
            rotate_query,
            sign_bytes_padded,
        )

        dev = kv_cache.device
        key_cache, value_cache = kv_cache.unbind(1)

        # Rotate query by K's rotation matrix (external, not in kernel)
        k_cb = self._k_codebook.to(dev)
        v_cb = self._v_codebook.to(dev)
        k_R_T = k_cb.rotation_matrix_T
        q_rot = rotate_query(query[:num_actual_tokens], k_R_T)

        # Fused attention: unified_attention reads compressed cache inline,
        # looks up centroids, applies norms and QJL correction — no staging.
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )

        unified_attention(
            q=q_rot,
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            kv_quant_mode=self._kv_quant_mode,
            tq_centroids=k_cb.centroids,
            tq_v_centroids=v_cb.centroids,
            tq_k_norms=self._k_norms,
            tq_v_norms=self._v_norms,
            tq_k_res_scales=(
                self._k_res_scales if self._preset.qjl else None
            ),
            tq_v_res_scales=(
                self._v_res_scales if self._preset.qjl else None
            ),
        )

        # Inverse-rotate output (V was in rotated space)
        v_R = v_cb.rotation_matrix
        out_slice = output[:num_actual_tokens]
        out_slice.copy_(inverse_rotate_output(out_slice, v_R))

        return output

    def _forward_outlier(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        output: torch.Tensor,
        num_actual_tokens: int,
    ) -> torch.Tensor:
        """Outlier mode: dequant to staging, then standard attention.

        No Q rotation or output inverse-rotation needed — the staging
        buffer contains data in the original (unrotated) space.

        Uses compacted staging: dequant writes sequentially to
        staging[seq_idx * max_blocks + block_pos], and unified_attention
        uses a sequential block_table to match.
        """
        from vllm.v1.attention.ops.turboquant import outlier_dequant_to_staging
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )

        dev = kv_cache.device
        num_seqs = attn_metadata.block_table.shape[0]
        max_blocks_per_seq = attn_metadata.block_table.shape[1]
        n_staging_blocks = num_seqs * max_blocks_per_seq

        self._ensure_staging_buffers(kv_cache, n_staging_blocks)
        key_cache, value_cache = kv_cache.unbind(1)

        # Sequential block table for staging: staging_bt[s, b] = s * max_blocks + b
        # The dequant kernel writes to staging_blk = seq_idx * max_blocks + block_pos,
        # so this mapping lets unified_attention find the right staging blocks.
        staging_bt = torch.arange(
            n_staging_blocks, device=dev, dtype=attn_metadata.block_table.dtype,
        ).reshape(num_seqs, max_blocks_per_seq)

        # Dequant K to staging (original space)
        k_cfg = self._k_outlier_config.to(dev)
        outlier_dequant_to_staging(
            key_cache, self._k_out_norms, self._k_reg_norms,
            self._staging_k, k_cfg,
            attn_metadata.block_table, attn_metadata.seq_lens,
            max_blocks_per_seq,
            out_rotated_buf=self._out_rotated_buf,
            reg_rotated_buf=self._reg_rotated_buf,
        )

        # Dequant V to staging (original space)
        v_cfg = self._v_outlier_config.to(dev)
        outlier_dequant_to_staging(
            value_cache, self._v_out_norms, self._v_reg_norms,
            self._staging_v, v_cfg,
            attn_metadata.block_table, attn_metadata.seq_lens,
            max_blocks_per_seq,
            out_rotated_buf=self._out_rotated_buf,
            reg_rotated_buf=self._reg_rotated_buf,
        )

        # Standard attention on bf16 staging with sequential block table
        unified_attention(
            q=query[:num_actual_tokens],
            k=self._staging_k,
            v=self._staging_v,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=staging_bt,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            kv_quant_mode=KVQuantMode.NONE,
        )

        return output

    # ----- Outlier Calibration -----

    def _calibrate_outlier_channels(
        self, key: torch.Tensor, value: torch.Tensor,
    ) -> None:
        """Calibrate outlier channel selection from actual K/V data.

        Computes per-channel variance across the batch and selects the
        top outlier_ratio fraction as outlier channels. Recreates the
        OutlierChannelConfig with the calibrated mask.

        This runs ONCE on the first forward pass, then the mask is frozen.
        """
        from vllm.v1.attention.ops.turboquant import OutlierChannelConfig

        # key/value: (num_tokens, num_kv_heads, head_dim)
        # Compute per-channel variance across tokens and heads
        k_float = key.float()
        v_float = value.float()
        # Combine K and V variance for a unified mask
        kv_cat = torch.cat([k_float, v_float], dim=0)  # (2T, H, D)
        channel_var = kv_cat.var(dim=(0, 1))  # (D,)

        outlier_dim = int(self.head_size * self._preset.outlier_ratio)
        # Select top-k channels by variance
        _, top_indices = channel_var.topk(outlier_dim)
        outlier_mask = torch.zeros(
            self.head_size, dtype=torch.bool, device=key.device,
        )
        outlier_mask[top_indices] = True

        logger.info(
            "TurboQuant outlier calibration: top-%d channels by variance "
            "(var range: [%.4f, %.4f], outlier mean var: %.4f, "
            "regular mean var: %.4f)",
            outlier_dim,
            channel_var.min().item(),
            channel_var.max().item(),
            channel_var[outlier_mask].mean().item(),
            channel_var[~outlier_mask].mean().item(),
        )

        # Recreate configs with calibrated mask
        self._k_outlier_config = OutlierChannelConfig(
            head_dim=self.head_size,
            outlier_ratio=self._preset.outlier_ratio,
            outlier_bits=self._preset.k_bits,
            regular_bits=self._preset.v_bits,
            seed=42,
            device=key.device.type,
            outlier_mask=outlier_mask.cpu(),
        )
        self._v_outlier_config = OutlierChannelConfig(
            head_dim=self.head_size,
            outlier_ratio=self._preset.outlier_ratio,
            outlier_bits=self._preset.k_bits,
            regular_bits=self._preset.v_bits,
            seed=43,
            device=key.device.type,
            outlier_mask=outlier_mask.cpu(),
        )

    # ----- KV Cache Update -----

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self._norms_dirty:
            self._norms_dirty = False
            self._reset_cache_views()
        self._ensure_cache_views(kv_cache)

        if self._outlier_mode and self._needs_calibration:
            self._calibrate_outlier_channels(key, value)
            self._needs_calibration = False

        key_cache, value_cache = kv_cache.unbind(1)
        dev = key.device

        if self._outlier_mode:
            from vllm.v1.attention.ops.turboquant import outlier_encode_single

            k_cfg = self._k_outlier_config.to(dev)
            outlier_encode_single(
                key, key_cache,
                self._k_out_norms, self._k_reg_norms,
                slot_mapping, k_cfg,
            )

            v_cfg = self._v_outlier_config.to(dev)
            outlier_encode_single(
                value, value_cache,
                self._v_out_norms, self._v_reg_norms,
                slot_mapping, v_cfg,
            )
        else:
            from vllm.v1.attention.ops.turboquant import (
                turboquant_encode_single,
            )

            assert self._k_norms is not None
            assert self._v_norms is not None

            k_cb = self._k_codebook.to(dev)
            turboquant_encode_single(
                key, key_cache, self._k_norms, slot_mapping, k_cb,
                signs_cache=self._k_signs,
                res_scales_cache=self._k_res_scales,
            )

            v_cb = self._v_codebook.to(dev)
            turboquant_encode_single(
                value, value_cache, self._v_norms, slot_mapping, v_cb,
                signs_cache=self._v_signs,
                res_scales_cache=self._v_res_scales,
            )
