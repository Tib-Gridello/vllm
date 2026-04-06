# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone TurboQuant KV Cache Compression Backend.

Implements PolarQuant from "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate" (arXiv:2504.19874).

Unique features over other TQ implementations:
- Hadamard rotation (3pp MMLU improvement over random QR)
- Sign correction (QJL variant, 1-bit residual per coordinate)
- Outlier channel support (for sub-4-bit, the paper's sweet spot)
- Dequant-first architecture (decompress → standard attention)
- Named presets via --kv-cache-dtype (e.g. tq-k8v8-qjl)

Usage:
  vllm serve <model> --kv-cache-dtype tq-k8v8-qjl
  vllm serve <model> --kv-cache-dtype tq-k4v4 --enforce-eager
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
    """TurboQuant attention: dequant-first for all paths.

    Encode (do_kv_cache_update):
      K/V → normalize → Hadamard rotate → quantize → pack → scatter

    Decode (forward):
      Cache → Triton dequant → bf16 staging → rotate Q → attention → inv-rotate
    """

    # Lazy-initialized state
    _k_norms: torch.Tensor | None = None
    _v_norms: torch.Tensor | None = None
    _k_signs: torch.Tensor | None = None
    _v_signs: torch.Tensor | None = None
    _k_res_scales: torch.Tensor | None = None
    _v_res_scales: torch.Tensor | None = None
    _staging_key: torch.Tensor | None = None
    _staging_val: torch.Tensor | None = None
    _staging_block_table: torch.Tensor | None = None
    _max_blocks_per_seq: int = 0
    _norms_dirty: bool = False

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

        # Create codebooks (separate for K and V to allow asymmetric bits)
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
            seed=43,  # different rotation for V
            device="cpu",
            qjl=self._preset.qjl,
        )

        logger.info(
            "TurboQuant backend: preset=%s, k_bits=%d, v_bits=%d, "
            "qjl=%s, head_dim=%d, padded_dim=%d",
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
        Layout per head: [indices | L2 norm (4B) | sign bits | res_scale (4B)]
        """
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

                # Uint8 view for sign bits
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

    def _reset_cache_views(self) -> None:
        self._k_norms = None
        self._v_norms = None
        self._k_signs = None
        self._v_signs = None
        self._k_res_scales = None
        self._v_res_scales = None

    def _ensure_staging(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Allocate bf16 staging buffers for dequant-first path."""
        num_seqs = block_table.shape[0]
        block_size = kv_cache.shape[2]
        nkv = kv_cache.shape[3]

        max_seq = int(seq_lens.max().item()) if num_seqs > 0 else 0
        max_bps = (max_seq + block_size - 1) // block_size
        if max_bps < 1:
            max_bps = 1
        needed = num_seqs * max_bps

        if (
            self._staging_key is not None
            and self._staging_key.shape[0] >= needed
            and self._max_blocks_per_seq >= max_bps
        ):
            return

        head_dim = self.head_size
        device = kv_cache.device

        self._max_blocks_per_seq = max_bps
        self._staging_key = torch.zeros(
            (needed, block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self._staging_val = torch.zeros(
            (needed, block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self._staging_block_table = torch.arange(
            0,
            needed,
            dtype=block_table.dtype,
            device=device,
        ).reshape(num_seqs, max_bps)

    # ----- Forward -----

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

        # --- Decompress TQ cache → bf16 staging ---
        from vllm.v1.attention.ops.turboquant import (
            inverse_rotate_output,
            rotate_query,
            turboquant_dequant_single,
        )

        self._ensure_staging(
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
        )

        dev = kv_cache.device
        key_cache, value_cache = kv_cache.unbind(1)

        assert self._k_norms is not None
        assert self._v_norms is not None
        assert self._staging_key is not None
        assert self._staging_val is not None
        assert self._staging_block_table is not None

        k_cb = self._k_codebook.to(dev)
        turboquant_dequant_single(
            key_cache,
            self._k_norms,
            self._staging_key,
            k_cb,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self._max_blocks_per_seq,
            signs=self._k_signs if k_cb.qjl else None,
            res_scales=self._k_res_scales if k_cb.qjl else None,
        )

        v_cb = self._v_codebook.to(dev)
        turboquant_dequant_single(
            value_cache,
            self._v_norms,
            self._staging_val,
            v_cb,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self._max_blocks_per_seq,
            signs=self._v_signs if v_cb.qjl else None,
            res_scales=self._v_res_scales if v_cb.qjl else None,
        )

        # Rotate query by K's rotation matrix
        k_R_T = k_cb.rotation_matrix_T
        q_rot = rotate_query(query[:num_actual_tokens], k_R_T)

        # Use staging block table for attention
        num_seqs = attn_metadata.block_table.shape[0]
        staging_bt = self._staging_block_table[:num_seqs]

        # Standard Triton unified attention on decompressed bf16 data
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )

        unified_attention(
            q=q_rot,
            k=self._staging_key,
            v=self._staging_val,
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

        # Inverse-rotate output (V was in rotated space)
        v_R = v_cb.rotation_matrix
        out_slice = output[:num_actual_tokens]
        out_slice.copy_(inverse_rotate_output(out_slice, v_R))

        return output

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

        from vllm.v1.attention.ops.turboquant import turboquant_encode_single

        key_cache, value_cache = kv_cache.unbind(1)
        dev = key.device

        assert self._k_norms is not None
        assert self._v_norms is not None

        k_cb = self._k_codebook.to(dev)
        turboquant_encode_single(
            key,
            key_cache,
            self._k_norms,
            slot_mapping,
            k_cb,
            signs_cache=self._k_signs,
            res_scales_cache=self._k_res_scales,
        )

        v_cb = self._v_codebook.to(dev)
        turboquant_encode_single(
            value,
            value_cache,
            self._v_norms,
            slot_mapping,
            v_cb,
            signs_cache=self._v_signs,
            res_scales_cache=self._v_res_scales,
        )
