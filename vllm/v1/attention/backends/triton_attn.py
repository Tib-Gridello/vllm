# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import (
    get_dtype_size,
    is_quantized_kv_cache,
    is_turboquant_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    kv_cache_uses_per_token_head_scales,
)

logger = init_logger(__name__)


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None

    @property
    def mm_prefix_range_tensor(self) -> torch.Tensor | None:
        """Convert mm_prefix_range dict to padded tensor for Triton kernel.

        Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
        Empty ranges have start==end==0, which kernel skips via is_valid check.
        """
        # TODO(Isotr0py): Move to model runner's attention metadata
        # preparation to avoid duplicate computation.
        if self.mm_prefix_range is None:
            return None

        num_seqs = self.seq_lens.shape[0]
        device = self.seq_lens.device

        # Collect ranges, using [(0,0)] for empty sequences to ensure uniform dims
        range_lists = [
            self.mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # Return None if all ranges are trivial (only (0,0) placeholders)
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # Create 2D tensors with shape (num_ranges, 2) for each sequence
        range_tensors = [
            torch.tensor(r, dtype=torch.int32, device=device).view(-1, 2)
            for r in range_lists
        ]

        return torch.nested.nested_tensor(
            range_tensors, layout=torch.jagged
        ).to_padded_tensor(0)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
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
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8_per_token_head",
        "fp8_per_token_head",
        "turboquant",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if is_turboquant_kv_cache(cache_dtype_str):
            import vllm.envs as envs
            scale_pad = get_dtype_size(torch.float32)  # 4 uint8 = 1 float32

            # Outlier mode: mixed-precision cache layout
            outlier_str = envs.VLLM_TURBOQUANT_OUTLIER_BITS
            if outlier_str:
                from vllm.v1.attention.ops.turboquant import (
                    OutlierChannelConfig,
                )
                parts = outlier_str.split(",")
                oc = OutlierChannelConfig(
                    head_dim=head_size,
                    outlier_bits=int(parts[0]),
                    regular_bits=int(parts[1]),
                    device="cpu",
                )
                padded = oc.cache_bytes_per_head()
                return (num_blocks, 2, block_size, num_kv_heads,
                        padded)

            bits = envs.VLLM_TURBOQUANT_BITS
            if bits > 0 and bits <= 4:
                # Nibble-packed: head_size//2 data bytes + 4 for norm
                data_dim = head_size // 2
                if envs.VLLM_TURBOQUANT_QJL:
                    from vllm.v1.attention.ops.turboquant import (
                        sign_bytes_padded,
                    )
                    qjl_pad = sign_bytes_padded(head_size) + scale_pad
                    return (num_blocks, 2, block_size, num_kv_heads,
                            data_dim + scale_pad + qjl_pad)
                return (num_blocks, 2, block_size, num_kv_heads,
                        data_dim + scale_pad)
            else:
                # Byte storage: head_size data bytes + 4 for norm
                data_dim = head_size
                if envs.VLLM_TURBOQUANT_QJL:
                    # QJL: also store sign bits + res_scale
                    from vllm.v1.attention.ops.turboquant import (
                        sign_bytes_padded,
                    )
                    qjl_pad = sign_bytes_padded(head_size) + scale_pad
                    return (num_blocks, 2, block_size, num_kv_heads,
                            data_dim + scale_pad + qjl_pad)
                return (num_blocks, 2, block_size, num_kv_heads,
                        data_dim + scale_pad)
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            # Pad head_size by sizeof(float32)/sizeof(cache_dtype) so
            # the per-head scale fits inline.  The backend extracts
            # data[:head_size] and scale[head_size:] via typed views.
            from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
            scale_pad = get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)
            return (num_blocks, 2, block_size, num_kv_heads, head_size + scale_pad)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, 2, num_kv_heads, num_layers, block_size, head_size)
            return (1, 2, 4, 0, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout: {cache_layout}")
        return stride_order

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    # Per-token-head quant: scale views carved from inline head padding.
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None
    # TurboQuant: norm views carved from inline head padding (same pattern).
    _tq_k_norms: torch.Tensor | None = None
    _tq_v_norms: torch.Tensor | None = None
    # QJL: sign-bit and residual-scale views.
    _tq_k_signs: torch.Tensor | None = None
    _tq_v_signs: torch.Tensor | None = None
    _tq_k_res_scales: torch.Tensor | None = None
    _tq_v_res_scales: torch.Tensor | None = None
    # Dequant-first staging buffers (bf16, sized for active batch)
    _tq_dequant_first: bool = False
    _tq_staging_key: torch.Tensor | None = None
    _tq_staging_val: torch.Tensor | None = None
    _tq_staging_block_table: torch.Tensor | None = None
    _tq_max_blocks_per_seq: int = 0
    # Incremental dequant: per-physical-block dirty flags.
    # When set, only dirty blocks are decompressed; clean blocks keep
    # their staging data from the previous step.
    _tq_dirty_blocks: torch.Tensor | None = None
    _tq_staging_valid: bool = False  # True after first full dequant

    def _ensure_tq_norm_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head norm/sign/res_scale views from padded cache dim.

        The KV cache shape is
        ``(num_blocks, 2, block_size, nkv, padded_dim)`` where:

        * **byte mode (no QJL):** ``padded_dim = head_size + 4`` —
          the last 4 uint8 bytes of each head hold one float32 norm.
        * **byte mode + QJL:** ``padded_dim = head_size + 4 + sign_bytes + 4``
          — layout is ``[indices | L2 norm | sign bits | res_scale]``.

        Creates float32 strided views for norms and (when QJL) res_scales,
        and a uint8 strided view for sign bytes.
        """
        if self._tq_k_norms is not None:
            return

        num_blocks, _, block_size, nkv, padded_dim = kv_cache.shape
        dtype_sz = kv_cache.element_size()  # 1 for uint8

        # Determine head_size from codebook (set during __init__)
        head_size = self._tq_codebook.head_dim
        qjl = self._tq_codebook.qjl

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor(
            [], dtype=torch.float32, device=kv_cache.device
        ).set_(raw)

        kv_half_bytes = block_size * nkv * padded_dim * dtype_sz
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim * dtype_sz // 4
        head_f32 = padded_dim * dtype_sz // 4
        # L2 norm sits right after the index data
        # Index data size: head_size for byte mode, head_size//2 for nibble
        idx_data_bytes = head_size if self._tq_codebook.byte_mode else head_size // 2
        norm_off_f32 = idx_data_bytes * dtype_sz // 4

        # K norms: kv_half=0
        self._tq_k_norms = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=norm_off_f32,
        )

        # V norms: kv_half=1
        v_base_f32 = kv_half_bytes // 4
        self._tq_v_norms = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + norm_off_f32,
        )

        # Zero norms — critical because the profiling/warmup pass may have
        # written stale data to these padding bytes.
        self._tq_k_norms.fill_(0.0)
        self._tq_v_norms.fill_(0.0)

        # QJL views: sign bits (uint8) and res_scale (float32)
        if qjl:
            from vllm.v1.attention.ops.turboquant import sign_bytes_padded
            sbytes = sign_bytes_padded(head_size)

            # --- Sign bits: uint8 view ---
            # Offset in bytes from start of each head's data:
            sign_byte_off = idx_data_bytes + 4  # after indices + L2 norm
            base_u8 = torch.tensor(
                [], dtype=torch.uint8, device=kv_cache.device
            ).set_(raw)
            kv_half_bytes_u8 = kv_half_bytes
            full_block_u8 = 2 * kv_half_bytes_u8
            slot_u8 = nkv * padded_dim
            head_u8 = padded_dim

            # K signs (kv_half=0)
            self._tq_k_signs = torch.as_strided(
                base_u8,
                size=(num_blocks, block_size, nkv, sbytes),
                stride=(full_block_u8, slot_u8, head_u8, 1),
                storage_offset=sign_byte_off,
            )
            # V signs (kv_half=1)
            self._tq_v_signs = torch.as_strided(
                base_u8,
                size=(num_blocks, block_size, nkv, sbytes),
                stride=(full_block_u8, slot_u8, head_u8, 1),
                storage_offset=kv_half_bytes_u8 + sign_byte_off,
            )
            self._tq_k_signs.fill_(0)
            self._tq_v_signs.fill_(0)

            # --- Residual scale: float32 view ---
            res_scale_byte_off = idx_data_bytes + 4 + sbytes
            res_scale_off_f32 = res_scale_byte_off // 4

            self._tq_k_res_scales = torch.as_strided(
                base_f32,
                size=(num_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=res_scale_off_f32,
            )
            self._tq_v_res_scales = torch.as_strided(
                base_f32,
                size=(num_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=v_base_f32 + res_scale_off_f32,
            )
            self._tq_k_res_scales.fill_(0.0)
            self._tq_v_res_scales.fill_(0.0)

    def _reset_tq_norms(self) -> None:
        """Invalidate norm views so they are recreated and re-zeroed.

        The profiling pass may write stale data to the cache's norm padding.
        Simply filling may not work if the storage changed, so we force view
        recreation by setting the cached views to None.
        """
        self._tq_k_norms = None
        self._tq_v_norms = None
        self._tq_k_signs = None
        self._tq_v_signs = None
        self._tq_k_res_scales = None
        self._tq_v_res_scales = None
        # Invalidate incremental dequant state
        self._tq_dirty_blocks = None
        self._tq_staging_valid = False

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded head dimension.

        The KV cache shape is ``(num_blocks, 2, block_size, nkv, hs+pad)``
        where ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The last
        ``pad`` elements of each head hold one float32 scale.  We create
        strided float32 views over those bytes.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        if self._k_scale_cache is not None:
            return
        from vllm.utils.torch_utils import get_dtype_size

        num_blocks, _, block_size, nkv, padded_hs = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz  # e.g. 4
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        # In the raw bytes, each (block, kv_half, slot, head) occupies
        # padded_hs * dtype_sz bytes.  The scale float32 sits at byte
        # offset hs * dtype_sz within that region.
        kv_half_bytes = block_size * nkv * padded_hs * dtype_sz
        full_block_f32 = 2 * kv_half_bytes // 4  # stride between blocks
        slot_f32 = nkv * padded_hs * dtype_sz // 4  # stride between slots
        head_f32 = padded_hs * dtype_sz // 4  # stride between heads
        scale_off_f32 = hs * dtype_sz // 4  # offset to scale within head

        # K scales: kv_half=0
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=scale_off_f32,
        )
        self._k_scale_cache.fill_(1.0)

        # V scales: kv_half=1, offset by kv_half_bytes
        v_base_f32 = kv_half_bytes // 4
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + scale_off_f32,
        )
        self._v_scale_cache.fill_(1.0)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

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
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.supports_quant_query_input = current_platform.is_cuda()

        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head
        self._is_turboquant = self._kv_quant_mode.is_turboquant
        self._tq_outlier_config = None
        if self._is_turboquant:
            import vllm.envs as envs
            from vllm.v1.attention.ops.turboquant import TurboQuantCodebook
            bits = envs.VLLM_TURBOQUANT_BITS
            if bits == 0:  # auto-select based on head dimension
                bits = 4 if head_size >= 256 else 8
            qjl = envs.VLLM_TURBOQUANT_QJL
            self._tq_codebook = TurboQuantCodebook(
                n_bits=bits,
                head_dim=head_size,
                device="cpu",  # moved to GPU on first use
                qjl=qjl,
            )
            self._tq_dequant_first = envs.VLLM_TURBOQUANT_DEQUANT_FIRST

            # Outlier channel mixed-precision (e.g., "4,2" for 2.5-bit)
            outlier_str = envs.VLLM_TURBOQUANT_OUTLIER_BITS
            if outlier_str:
                from vllm.v1.attention.ops.turboquant import (
                    OutlierChannelConfig,
                )
                parts = outlier_str.split(",")
                out_bits = int(parts[0])
                reg_bits = int(parts[1])
                self._tq_outlier_config = OutlierChannelConfig(
                    head_dim=head_size,
                    outlier_ratio=0.25,
                    outlier_bits=out_bits,
                    regular_bits=reg_bits,
                    device="cpu",
                    qjl=False,  # QJL off for mixed-precision
                )

    def _ensure_tq_staging(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
    ) -> None:
        """Lazily allocate bf16 staging buffers for dequant-first path.

        Staging layout: [num_seqs * max_bps, block_size, nkv, head_dim]
        Block table remap: staging_bt[s, p] = s * max_bps + p

        Uses actual max(seq_lens) to compute max_bps rather than the
        block_table width (which reflects max_model_len and can be huge).
        """
        _, _, block_size_cache, nkv, _ = kv_cache.shape

        # Compute max_blocks_per_seq from actual sequence lengths
        bt_width = block_table.shape[1]
        if seq_lens is not None and seq_lens.numel() > 0:
            max_sl = seq_lens.max().item()
            max_bps = min(
                (max_sl + block_size_cache - 1) // block_size_cache,
                bt_width,
            )
        else:
            max_bps = bt_width
        # Ensure at least 1 block per seq
        max_bps = max(max_bps, 1)
        num_seqs = block_table.shape[0]
        needed = num_seqs * max_bps

        # Check if staging is already big enough
        if (self._tq_staging_key is not None
                and self._tq_staging_key.shape[0] >= needed
                and self._tq_max_blocks_per_seq == max_bps):
            # Rebuild staging block_table if num_seqs changed
            if (self._tq_staging_block_table is None
                    or self._tq_staging_block_table.shape[0] < num_seqs):
                self._tq_staging_block_table = torch.arange(
                    0, needed,
                    dtype=block_table.dtype,
                    device=block_table.device,
                ).reshape(num_seqs, max_bps)
            return

        head_dim = self._tq_codebook.head_dim
        device = kv_cache.device

        self._tq_max_blocks_per_seq = max_bps
        self._tq_staging_key = torch.zeros(
            (needed, block_size_cache, nkv, head_dim),
            dtype=torch.bfloat16, device=device,
        )
        self._tq_staging_val = torch.zeros(
            (needed, block_size_cache, nkv, head_dim),
            dtype=torch.bfloat16, device=device,
        )
        self._tq_staging_block_table = torch.arange(
            0, needed,
            dtype=block_table.dtype, device=device,
        ).reshape(num_seqs, max_bps)
        # Staging was reallocated → force full dequant on next forward
        self._tq_staging_valid = False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.  Mark that norms need clearing before first
            # real inference (warmup writes stale data to cache padding).
            if self._is_turboquant:
                self._tq_norms_dirty = True
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # TurboQuant: packed uint8 indices with inline float32 norms.
        tq_centroids = None
        tq_rotation_signs = None
        tq_k_norms = None
        tq_v_norms = None
        tq_k_res_scales = None
        tq_v_res_scales = None
        kv_quant_mode_for_attn = self._kv_quant_mode
        _tq_skip_output_rotation = False
        if self._is_turboquant:
            self._ensure_tq_norm_caches(kv_cache)
            cb = self._tq_codebook
            dev = kv_cache.device
            k_descale = None
            v_descale = None
            k_scale_cache = None
            v_scale_cache = None

            if self._tq_outlier_config is not None:
                # --- OUTLIER MIXED-PRECISION PATH ---
                # Dequant outputs in ORIGINAL space → no rotation needed
                from vllm.v1.attention.ops.turboquant import (
                    outlier_dequant_paged,
                )
                self._ensure_tq_staging(
                    kv_cache, attn_metadata.block_table,
                    attn_metadata.seq_lens)
                key_cache_tq, value_cache_tq = kv_cache.unbind(1)
                outlier_dequant_paged(
                    key_cache_tq, value_cache_tq,
                    self._tq_k_norms, self._tq_v_norms,
                    self._tq_staging_key, self._tq_staging_val,
                    self._tq_outlier_config.to(dev),
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    self._tq_max_blocks_per_seq,
                )
                key_cache = self._tq_staging_key
                value_cache = self._tq_staging_val
                kv_quant_mode_for_attn = KVQuantMode.NONE
                _tq_skip_output_rotation = True
                # No Q rotation needed (KV in original space)
                query = query[:num_actual_tokens]
            else:
                # --- UNIFORM BIT-WIDTH TQ ---
                # Rotate Q by R (common to fused and dequant-first)
                from vllm.v1.attention.ops.turboquant import rotate_query
                R_T = cb.rotation_matrix_T.to(dev, non_blocking=True)
                query = rotate_query(query[:num_actual_tokens], R_T)

                # Auto-select dequant-first vs fused based on context
                use_dequant_first = (
                    self._tq_dequant_first
                    and attn_metadata.max_seq_len > 256
                )

                if use_dequant_first:
                    # Decompress TQ → bf16 staging, standard attention.
                    # Incremental: only dirty blocks are decompressed.
                    from vllm.v1.attention.ops.turboquant import (
                        turboquant_dequant_paged,
                    )
                    self._ensure_tq_staging(
                        kv_cache, attn_metadata.block_table,
                        attn_metadata.seq_lens)
                    key_cache_tq, value_cache_tq = kv_cache.unbind(1)
                    turboquant_dequant_paged(
                        key_cache_tq, value_cache_tq,
                        self._tq_k_norms, self._tq_v_norms,
                        self._tq_staging_key, self._tq_staging_val,
                        cb,
                        attn_metadata.block_table,
                        attn_metadata.seq_lens,
                        self._tq_max_blocks_per_seq,
                        k_signs=(self._tq_k_signs
                                 if cb.qjl else None),
                        v_signs=(self._tq_v_signs
                                 if cb.qjl else None),
                        k_res_scales=(self._tq_k_res_scales
                                      if cb.qjl else None),
                        v_res_scales=(self._tq_v_res_scales
                                      if cb.qjl else None),
                        dirty_blocks=(self._tq_dirty_blocks
                                      if self._tq_staging_valid
                                      else None),
                    )
                    # Clear dirty flags after dequant
                    if self._tq_dirty_blocks is not None:
                        self._tq_dirty_blocks.fill_(False)
                        self._tq_staging_valid = True
                    key_cache = self._tq_staging_key
                    value_cache = self._tq_staging_val
                    kv_quant_mode_for_attn = KVQuantMode.NONE
                else:
                    # Fused inline dequant in attention kernel
                    key_cache, value_cache = kv_cache.unbind(1)
                    tq_centroids = cb.centroids.to(
                        dev, non_blocking=True)
                    tq_rotation_signs = None
                    tq_k_norms = self._tq_k_norms
                    tq_v_norms = self._tq_v_norms
                    if cb.qjl:
                        tq_k_res_scales = self._tq_k_res_scales
                        tq_v_res_scales = self._tq_v_res_scales
        # Per-token-head quantized KV cache: use separate scale caches.
        elif self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            k_descale = None
            v_descale = None
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
        # FP8 per-tensor / auto path (original flow).
        else:
            key_cache, value_cache = kv_cache.unbind(1)
            if is_quantized_kv_cache(self.kv_cache_dtype):
                if key_cache.dtype != self.fp8_dtype:
                    key_cache = key_cache.view(self.fp8_dtype)
                    value_cache = value_cache.view(self.fp8_dtype)
                assert layer._q_scale_float == 1.0, (
                    "A non 1.0 q_scale is not currently supported."
                )
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[2],
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        # Dequant-first / outlier mode uses remapped staging block_table
        _uses_staging = (
            self._is_turboquant and (
                self._tq_outlier_config is not None
                or (self._tq_dequant_first
                    and attn_metadata.max_seq_len > 256)
            )
        )
        if _uses_staging:
            num_seqs = attn_metadata.block_table.shape[0]
            block_table = self._tq_staging_block_table[:num_seqs]
        else:
            block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        # For TQ, query was already rotated and sliced above
        q_for_attn = query if self._is_turboquant else query[:num_actual_tokens]
        unified_attention(
            q=q_for_attn,
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            kv_quant_mode=kv_quant_mode_for_attn,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            tq_centroids=tq_centroids,
            tq_rotation_signs=tq_rotation_signs,
            tq_k_norms=tq_k_norms,
            tq_v_norms=tq_v_norms,
            tq_k_res_scales=tq_k_res_scales,
            tq_v_res_scales=tq_v_res_scales,
        )

        # TQ: inverse-rotate attention output (V was in rotated space)
        # Skip for outlier mode (dequant already outputs in original space)
        if self._is_turboquant and not _tq_skip_output_rotation:
            from vllm.v1.attention.ops.turboquant import inverse_rotate_output
            R = self._tq_codebook.rotation_matrix.to(output.device)
            out_slice = output[:num_actual_tokens]
            out_slice.copy_(inverse_rotate_output(out_slice, R))

        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # Quantized KV cache is not supported for encoder attention.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantized KV cache is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # Reshape the input keys and values and store them in the cache.
        if self._is_turboquant:
            # Invalidate norm views BEFORE recreating if dirty (warmup wrote
            # stale data; views may point to old storage).
            if getattr(self, '_tq_norms_dirty', False):
                self._tq_norms_dirty = False
                self._reset_tq_norms()  # sets _tq_k_norms = None
            self._ensure_tq_norm_caches(kv_cache)  # recreates + fills 0
            key_cache, value_cache = kv_cache.unbind(1)

            if self._tq_outlier_config is not None:
                from vllm.v1.attention.ops.turboquant import (
                    outlier_reshape_and_cache,
                )
                outlier_reshape_and_cache(
                    key, value,
                    key_cache, value_cache,
                    self._tq_k_norms, self._tq_v_norms,
                    slot_mapping,
                    self._tq_outlier_config.to(key.device),
                )
            else:
                from vllm.v1.attention.ops.turboquant import (
                    turboquant_reshape_and_cache,
                )
                turboquant_reshape_and_cache(
                    key, value,
                    key_cache, value_cache,
                    self._tq_k_norms, self._tq_v_norms,
                    slot_mapping, self._tq_codebook.to(key.device),
                    k_signs=self._tq_k_signs,
                    v_signs=self._tq_v_signs,
                    k_res_scales=self._tq_k_res_scales,
                    v_res_scales=self._tq_v_res_scales,
                )

            # Mark written blocks as dirty for incremental dequant.
            if self._tq_dequant_first:
                num_blocks = kv_cache.shape[0]
                block_size = kv_cache.shape[2]
                if self._tq_dirty_blocks is None:
                    self._tq_dirty_blocks = torch.ones(
                        num_blocks, dtype=torch.bool,
                        device=kv_cache.device)
                    self._tq_staging_valid = False
                elif self._tq_dirty_blocks.shape[0] < num_blocks:
                    # Grow array preserving old state; only new blocks
                    # are marked dirty (old blocks may already be staged).
                    old = self._tq_dirty_blocks
                    self._tq_dirty_blocks = torch.ones(
                        num_blocks, dtype=torch.bool,
                        device=kv_cache.device)
                    self._tq_dirty_blocks[:old.shape[0]] = old
                    self._tq_staging_valid = False
                valid = slot_mapping >= 0
                if valid.any():
                    dirty_blks = slot_mapping[valid] // block_size
                    self._tq_dirty_blocks[dirty_blks] = True
            return
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                slot_mapping,
            )
            return
        # For decoder and cross-attention, use KV cache as before.
        key_cache, value_cache = kv_cache.unbind(1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(1)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
