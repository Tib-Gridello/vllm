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
- Exact Lloyd-Max codebook (pre-computed tables, Monte Carlo fallback)
- MSE-only by default (paper's Algorithm 1 for KV cache; QJL opt-in)
- Fused decode (no decompression to bf16 staging)
- Named presets via --kv-cache-dtype

Usage:
  vllm serve <model> --kv-cache-dtype tq_k8v8      # 8-bit, 2x compression
  vllm serve <model> --kv-cache-dtype tq_k4v4      # 4-bit, 4x compression
  vllm serve <model> --kv-cache-dtype tq_k8v8_qjl  # with sign correction (opt-in)
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

    # Prefill/decode split (populated by reorder_batch_threshold)
    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # 3D kernel parameters for parallel softmax segments (decode speedup)
    seq_threshold_3D: int = 0
    num_par_softmax_segments: int = 0
    softmax_segm_output: torch.Tensor | None = None
    softmax_segm_max: torch.Tensor | None = None
    softmax_segm_expsum: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Metadata Builder
# ---------------------------------------------------------------------------


class TurboQuantMetadataBuilder(
    AttentionMetadataBuilder[TurboQuantMetadata],
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # TurboQuant is not yet validated with speculative decoding.
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "TurboQuant KV cache compression does not support "
                "speculative decoding. Remove --speculative-model or "
                "use a different --kv-cache-dtype."
            )

        self.device = device
        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        from vllm.config import CUDAGraphMode
        from vllm.utils.math_utils import next_power_of_2

        # 3D kernel: parallelize decode across KV tiles for small batches.
        # Threshold below which the 3D kernel is used instead of 2D.
        MIN_LAUNCH_GRID_SIZE_2D = 128
        NUM_PAR_SOFTMAX_SEGMENTS = 16

        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        decode_cudagraph_enabled = vllm_config.compilation_config.cudagraph_mode in (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            CUDAGraphMode.FULL_DECODE_ONLY,
            CUDAGraphMode.FULL,
        )
        if decode_cudagraph_enabled:
            capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes
            if capture_sizes:
                self.seq_threshold_3D = min(
                    capture_sizes,
                    key=lambda x: abs(x - self.seq_threshold_3D),
                )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
            ),
            dtype=torch.float32,
            device=device,
        )

        # Enable prefill/decode batch reordering. Requests with
        # query_len <= 1 (decode) are sorted to the front of the batch,
        # enabling separate optimized code paths for each phase.
        self._init_reorder_batch_threshold(1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TurboQuantMetadata:
        from vllm.v1.attention.backends.utils import split_decodes_and_prefills

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata, decode_threshold=1)
        )

        return TurboQuantMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> TurboQuantMetadata:
        # CUDAGraph capture is always pure decode.
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        return TurboQuantMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=1,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_reqs,
            num_prefills=0,
            num_decode_tokens=num_tokens,
            num_prefill_tokens=0,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TurboQuantAttentionBackend(AttentionBackend):
    """Standalone TurboQuant KV cache compression backend.

    Compatible with: sliding window, GQA/MHA, prefix caching, chunked
    prefill, tensor parallelism, CUDA graphs.

    Not compatible with: speculative decoding (validated in
    TurboQuantMetadataBuilder.__init__), encoder/encoder-decoder models.

    For 4-bit presets (tq_k4v4, tq_k8fv4), use --kv-cache-dtype-skip-layers
    to keep the first/last layers in bf16 for quality.
    """

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
        cache_dtype_str: str = "tq_k8v8_qjl",
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
        # Triton encode/dequant kernels use tl.arange(0, HEAD_DIM) which
        # requires HEAD_DIM to be a power of 2. The block-diagonal
        # Hadamard rotation in generate_rotation_matrix() already
        # supports arbitrary sizes — this gate can be relaxed once the
        # Triton kernels adopt padded dims + masks.
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

    # Base seeds for K and V rotation matrices. Each layer gets a
    # UNIQUE rotation by adding ``layer_idx * _LAYER_SEED_STRIDE`` to
    # these bases. This is critical: if every layer used the same
    # rotation, quantization errors would be perfectly correlated
    # across layers and compound instead of averaging out. K and V
    # bases differ so their errors are statistically independent.
    _K_ROTATION_SEED_BASE = 42
    _V_ROTATION_SEED_BASE = 43
    _LAYER_SEED_STRIDE = 1337  # large prime; avoids overlap with K/V offset

    # Per-instance cache views (point into each layer's KV cache)
    _k_norms: torch.Tensor | None = None
    _v_norms: torch.Tensor | None = None
    _k_signs: torch.Tensor | None = None
    _v_signs: torch.Tensor | None = None
    _k_res_scales: torch.Tensor | None = None
    _v_res_scales: torch.Tensor | None = None
    _k_scales: torch.Tensor | None = None  # FP8 key per-token-head scales
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
        layer_name: str | None = None,
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
        self.layer_name = layer_name or ""

        # Derive per-layer rotation seeds from the layer name so every
        # layer has a statistically independent rotation matrix.
        # Parses indices from names like "model.layers.5.self_attn.attn".
        import regex as _re

        m = _re.search(r"layers[._](\d+)", self.layer_name)
        layer_idx = int(m.group(1)) if m else 0
        self._k_rotation_seed = (
            self._K_ROTATION_SEED_BASE + layer_idx * self._LAYER_SEED_STRIDE
        )
        self._v_rotation_seed = (
            self._V_ROTATION_SEED_BASE + layer_idx * self._LAYER_SEED_STRIDE
        )

        # Feature compatibility validation
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"TurboQuant only supports DECODER attention, got {attn_type}"
            )
        if head_size < 32 or (head_size & (head_size - 1)) != 0:
            raise ValueError(
                f"TurboQuant requires power-of-2 head_size >= 32, got "
                f"{head_size}. Non-power-of-2 support requires padded "
                f"Triton kernels (see supports_head_size docstring)."
            )

        if sliding_window is not None:
            self.sliding_window = (sliding_window, sliding_window)
        else:
            self.sliding_window = (-1, -1)

        # Parse preset
        self._preset = parse_tq_preset(kv_cache_dtype)

        if self._preset.k_fp8:
            # FP8 key mode: FP8 keys (near-lossless) + TQ nibble values.
            # K stored as fp8_e4m3 + per-token-head float32 scale.
            # V stored as TQ quantized indices + L2 norm.
            # No Q rotation needed (K is in original space).
            self._kv_quant_mode = KVQuantMode.TURBOQUANT_FP8_KEY

            from vllm.v1.attention.ops.turboquant import TurboQuantCodebook

            self._k_codebook = None  # No codebook for FP8 keys
            self._v_codebook = TurboQuantCodebook(
                n_bits=self._preset.v_bits,
                head_dim=head_size,
                seed=self._v_rotation_seed,
                device="cpu",
                qjl=self._preset.qjl,
            )

            # FP8 E4M3 decode LUT: 256 float values for all possible bytes.
            # Used by the Triton kernel as a gather-based FP8 decoder that
            # works on all GPU architectures (A100, H100, B100).
            # Kept on CPU; lazily cached on CUDA device in forward().
            fp8_bytes = torch.arange(256, dtype=torch.uint8)
            self._fp8_lut = fp8_bytes.view(torch.float8_e4m3fn).float()
            self._fp8_lut_device: torch.Tensor | None = None

            logger.info(
                "TurboQuant backend: preset=%s, k=FP8, v_bits=%d, "
                "qjl=%s, head_dim=%d, padded_dim=%d, fused=True",
                self._preset.name,
                self._preset.v_bits,
                self._preset.qjl,
                head_size,
                self._preset.padded_cache_dim(head_size),
            )
        else:
            # Standard mode: fused attention with inline dequant.
            k_byte = self._preset.k_byte_mode
            v_byte = self._preset.v_byte_mode
            if k_byte and not v_byte:
                self._kv_quant_mode = KVQuantMode.TURBOQUANT_MIXED
            elif k_byte:
                self._kv_quant_mode = KVQuantMode.TURBOQUANT_BYTE
            else:
                if not k_byte and v_byte:
                    raise NotImplementedError(
                        "Nibble K + byte V not supported. "
                        f"k_bits={self._preset.k_bits}, "
                        f"v_bits={self._preset.v_bits}."
                    )
                self._kv_quant_mode = KVQuantMode.TURBOQUANT

            from vllm.v1.attention.ops.turboquant import TurboQuantCodebook

            self._k_codebook = TurboQuantCodebook(
                n_bits=self._preset.k_bits,
                head_dim=head_size,
                seed=self._k_rotation_seed,
                device="cpu",
                qjl=self._preset.qjl,
            )
            self._v_codebook = TurboQuantCodebook(
                n_bits=self._preset.v_bits,
                head_dim=head_size,
                seed=self._v_rotation_seed,
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
        """
        if self._preset.k_fp8:
            if self._k_scales is not None:
                return
            self._ensure_fp8_key_cache_views(kv_cache)
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

        buf_numel_f32 = raw.nbytes() // 4

        # Process K and V with their respective codebooks
        for kv_idx, cb, attr_prefix in [
            (0, self._k_codebook, "_k"),
            (1, self._v_codebook, "_v"),
        ]:
            hd = cb.head_dim
            idx_bytes = hd if cb.byte_mode else hd // 2
            norm_off_f32 = idx_bytes // 4
            kv_offset_f32 = kv_idx * kv_half_bytes // 4

            off = kv_offset_f32 + norm_off_f32
            assert (
                off
                + (num_blocks - 1) * full_block_f32
                + (block_size - 1) * slot_f32
                + (nkv - 1) * head_f32
                < buf_numel_f32
            ), f"TQ norm view OOB: offset={off}, buf={buf_numel_f32}"

            norms = torch.as_strided(
                base_f32,
                size=(num_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=off,
            )
            norms.fill_(0.0)
            setattr(self, f"{attr_prefix}_norms", norms)

            if cb.qjl:
                from vllm.v1.attention.ops.turboquant import sign_bytes_padded

                sbytes = sign_bytes_padded(hd)

                # Float32 view for res_scale
                res_off_f32 = (idx_bytes + 4 + sbytes) // 4
                res_off = kv_offset_f32 + res_off_f32
                assert (
                    res_off
                    + (num_blocks - 1) * full_block_f32
                    + (block_size - 1) * slot_f32
                    + (nkv - 1) * head_f32
                    < buf_numel_f32
                ), f"TQ res_scale view OOB: offset={res_off}, buf={buf_numel_f32}"

                res_scales = torch.as_strided(
                    base_f32,
                    size=(num_blocks, block_size, nkv),
                    stride=(full_block_f32, slot_f32, head_f32),
                    storage_offset=res_off,
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

    def _ensure_fp8_key_cache_views(self, kv_cache: torch.Tensor) -> None:
        """Extract FP8 key scale views and V norm views from cache padding.

        FP8 key cache layout per head:
          [head_dim bytes fp8 data | 4 bytes float32 scale]
        V cache layout per head (standard TQ):
          [indices | L2 norm (4B)]
        """
        num_blocks, _, block_size, nkv, padded_dim = kv_cache.shape
        raw = kv_cache.untyped_storage()

        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        kv_half_bytes = block_size * nkv * padded_dim
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim // 4
        head_f32 = padded_dim // 4

        buf_numel_f32 = raw.nbytes() // 4
        last_elem = lambda off: (  # noqa: E731
            off
            + (num_blocks - 1) * full_block_f32
            + (block_size - 1) * slot_f32
            + (nkv - 1) * head_f32
        )

        # K half: FP8 scale at byte offset head_dim (after fp8 data)
        hd = self.head_size
        k_scale_off_f32 = hd // 4  # head_dim bytes / 4
        k_kv_offset_f32 = 0  # K is first half

        k_off = k_kv_offset_f32 + k_scale_off_f32
        assert last_elem(k_off) < buf_numel_f32, (
            f"FP8 K scale view OOB: offset={k_off}, buf={buf_numel_f32}"
        )

        k_scales = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=k_off,
        )
        k_scales.fill_(1.0)
        self._k_scales = k_scales

        # V half: standard TQ norm extraction
        v_cb = self._v_codebook
        v_idx_bytes = hd if v_cb.byte_mode else hd // 2
        v_norm_off_f32 = v_idx_bytes // 4
        v_kv_offset_f32 = kv_half_bytes // 4  # V is second half

        v_off = v_kv_offset_f32 + v_norm_off_f32
        assert last_elem(v_off) < buf_numel_f32, (
            f"FP8 V norm view OOB: offset={v_off}, buf={buf_numel_f32}"
        )

        v_norms = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_off,
        )
        v_norms.fill_(0.0)
        self._v_norms = v_norms

    def _reset_cache_views(self) -> None:
        self._k_norms = None
        self._v_norms = None
        self._k_signs = None
        self._v_signs = None
        self._k_res_scales = None
        self._v_res_scales = None
        self._k_scales = None

    # ----- Flash-attn prefill (first-chunk only) -----

    def _flash_attn_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Use flash_attn for first-chunk prefill (K/V still in bf16).

        During first-chunk prefill, max_query_len == max_seq_len — all
        tokens are new, no prior KV cache context. K and V haven't been
        encoded to TQ yet, so we can use flash_attn directly on the raw
        bf16 tensors. This is both faster (CUDA kernel) and lossless
        (no quantization noise on the prefill step).
        """
        from vllm.v1.attention.backends.fa_utils import (
            is_flash_attn_varlen_func_available,
        )

        if not is_flash_attn_varlen_func_available():
            # Fall back to SDPA if flash_attn not available
            return self._sdpa_prefill(query, key, value, attn_metadata, output)

        from vllm.v1.attention.backends.fa_utils import (
            flash_attn_varlen_func,
        )

        num_actual_tokens = attn_metadata.num_actual_tokens
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key[:num_actual_tokens],
            v=value[:num_actual_tokens],
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
        )
        return output

    def _sdpa_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback SDPA prefill when flash_attn is unavailable."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens]
        k = key[:num_actual_tokens]
        v = value[:num_actual_tokens]

        # Simple per-request SDPA (no paging needed for first-chunk)
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = len(attn_metadata.seq_lens)
        for i in range(num_reqs):
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            qi = q[start:end].unsqueeze(0).transpose(1, 2)
            ki = k[start:end].unsqueeze(0).transpose(1, 2)
            vi = v[start:end].unsqueeze(0).transpose(1, 2)
            oi = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, is_causal=True, scale=self.scale
            )
            output[start:end] = oi.squeeze(0).transpose(0, 1)
        return output

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

        # First-chunk prefill optimization: when all tokens are new
        # (max_query_len == max_seq_len, no prior KV cache context),
        # K/V are still in bf16. Use flash_attn directly — faster and
        # lossless (no quantization noise on the prefill step).
        is_pure_first_prefill = (
            attn_metadata.max_query_len > 1
            and attn_metadata.max_query_len == attn_metadata.max_seq_len
        )
        if is_pure_first_prefill:
            return self._flash_attn_prefill(query, key, value, attn_metadata, output)

        self._ensure_cache_views(kv_cache)

        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )
        from vllm.v1.attention.ops.turboquant import (
            inverse_rotate_output,
            rotate_query,
        )

        dev = kv_cache.device
        key_cache, value_cache = kv_cache.unbind(1)

        # Sparse V: skip V accumulation for tiles where max attention
        # weight is below threshold. Saves ~20-30% of inner loop work
        # per skipped tile. Bounded error: tile_size × threshold.
        #
        # Threshold tuning by context length:
        #   >= 8K context decode: aggressive (1e-3, ~10-15% speedup)
        #   >= 2K context decode: moderate  (1e-5, ~5-10% speedup)
        #   <  2K or prefill:     off (sparsity too low to help)
        max_ql = attn_metadata.max_query_len
        max_sl = attn_metadata.max_seq_len
        if max_ql == 1 and max_sl >= 8192:
            svt = 1e-3
        elif max_ql == 1 and max_sl >= 2048:
            svt = 1e-5
        else:
            svt = 0.0

        if self._preset.k_fp8:
            # FP8 key mode: no Q rotation (K is in original space).
            # K = FP8 dequant (cast + scale), V = TQ dequant (centroid lookup).
            # Eagerly move to device to avoid CPU→CUDA copies during
            # CUDAGraph capture (which forbids such copies).
            if self._fp8_lut_device is None or self._fp8_lut_device.device != dev:
                self._fp8_lut_device = self._fp8_lut.to(dev)
                self._v_codebook.to(dev)
            v_cb = self._v_codebook

            unified_attention(
                q=query[:num_actual_tokens],
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
                seq_threshold_3D=attn_metadata.seq_threshold_3D,
                num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
                softmax_segm_output=attn_metadata.softmax_segm_output,
                softmax_segm_max=attn_metadata.softmax_segm_max,
                softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                kv_quant_mode=self._kv_quant_mode,
                tq_centroids=self._fp8_lut_device,  # FP8 decode LUT (256 floats)
                tq_v_centroids=v_cb.centroids,
                tq_k_norms=self._k_scales,  # FP8 per-token-head scales
                tq_v_norms=self._v_norms,
                tq_k_res_scales=None,
                tq_v_res_scales=None,
                sparse_v_threshold=svt,
            )

            # Inverse-rotate output (V was in rotated space)
            v_R = v_cb.rotation_matrix
            out_slice = output[:num_actual_tokens]
            out_slice.copy_(inverse_rotate_output(out_slice, v_R))

            return output

        # Standard TQ mode: rotate Q by K's rotation matrix.
        # Eagerly move codebooks to device to avoid CPU→CUDA copies
        # during CUDAGraph capture.
        assert self._k_codebook is not None
        if self._k_codebook.centroids.device != dev:
            self._k_codebook.to(dev)
            self._v_codebook.to(dev)
        k_cb = self._k_codebook
        v_cb = self._v_codebook
        k_R_T = k_cb.rotation_matrix_T
        q_rot = rotate_query(query[:num_actual_tokens], k_R_T)

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
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            kv_quant_mode=self._kv_quant_mode,
            tq_centroids=k_cb.centroids,
            tq_v_centroids=v_cb.centroids,
            tq_k_norms=self._k_norms,
            tq_v_norms=self._v_norms,
            tq_k_res_scales=(self._k_res_scales if self._preset.qjl else None),
            tq_v_res_scales=(self._v_res_scales if self._preset.qjl else None),
            sparse_v_threshold=svt,
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

        key_cache, value_cache = kv_cache.unbind(1)
        dev = key.device

        if self._preset.k_fp8:
            # FP8 key mode: K → FP8 encode (no rotation), V → TQ encode
            from vllm.v1.attention.ops.turboquant import (
                fp8_encode_key,
                turboquant_encode_single,
            )

            assert self._k_scales is not None
            assert self._v_norms is not None

            fp8_encode_key(
                key,
                key_cache,
                self._k_scales,
                slot_mapping,
            )

            v_cb = self._v_codebook
            if v_cb.centroids.device != dev:
                v_cb = v_cb.to(dev)
            turboquant_encode_single(
                value,
                value_cache,
                self._v_norms,
                slot_mapping,
                v_cb,
                signs_cache=self._v_signs,
                res_scales_cache=self._v_res_scales,
            )
        else:
            from vllm.v1.attention.ops.turboquant import (
                turboquant_encode_single,
            )

            assert self._k_norms is not None
            assert self._v_norms is not None
            assert self._k_codebook is not None

            k_cb = self._k_codebook
            if k_cb.centroids.device != dev:
                k_cb = k_cb.to(dev)
            turboquant_encode_single(
                key,
                key_cache,
                self._k_norms,
                slot_mapping,
                k_cb,
                signs_cache=self._k_signs,
                res_scales_cache=self._k_res_scales,
            )

            v_cb = self._v_codebook
            if v_cb.centroids.device != dev:
                v_cb = v_cb.to(dev)
            turboquant_encode_single(
                value,
                value_cache,
                self._v_norms,
                slot_mapping,
                v_cb,
                signs_cache=self._v_signs,
                res_scales_cache=self._v_res_scales,
            )
