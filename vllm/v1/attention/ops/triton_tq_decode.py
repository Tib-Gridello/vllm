# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Specialized TurboQuant decode attention kernel.

Reads compressed KV cache (centroid indices + norms + QJL sign bits)
and computes attention in a single fused pass. Uses small BLOCK_KV=4
tiles to keep centroid lookup overhead manageable.

Uses split-KV pattern: stage1 computes partial attention per split,
stage2 (reused from triton_decode_attention) merges across splits.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tq_decode_stage1(
    # Query: (batch, num_q_heads, head_dim)
    Q,
    stride_q_b: tl.int64,
    stride_q_h: tl.int64,
    # Key cache: (num_blocks, block_size, num_kv_heads, padded_dim) uint8
    K_cache,
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    stride_kc_dim: tl.int64,
    # Value cache: same layout
    V_cache,
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_vc_dim: tl.int64,
    # Centroids: (n_levels,) float32
    Centroids,
    # K norms: (num_blocks, block_size, num_kv_heads) float32 strided
    K_norms,
    stride_kn_blk: tl.int64,
    stride_kn_slot: tl.int64,
    stride_kn_head: tl.int64,
    # V norms: same layout
    V_norms,
    stride_vn_blk: tl.int64,
    stride_vn_slot: tl.int64,
    stride_vn_head: tl.int64,
    # QJL K res_scales: (num_blocks, block_size, num_kv_heads) float32
    K_res_scales,
    stride_krs_blk: tl.int64,
    stride_krs_slot: tl.int64,
    stride_krs_head: tl.int64,
    # QJL V res_scales
    V_res_scales,
    stride_vrs_blk: tl.int64,
    stride_vrs_slot: tl.int64,
    stride_vrs_head: tl.int64,
    # Block table: (batch, max_blocks_per_seq) int32
    Block_table,
    stride_bt_b: tl.int64,
    stride_bt_pos: tl.int64,
    # Seq lens: (batch,) int32
    Seq_lens,
    # Output: (batch, num_q_heads, num_kv_splits, head_dim + 1)
    Att_Out,
    stride_ao_b: tl.int64,
    stride_ao_h: tl.int64,
    stride_ao_s: tl.int64,
    # Scalars
    sm_scale,
    # Constexprs
    kv_group_num: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    QJL_ENABLED: tl.constexpr,
    SIGN_DATA_OFFSET: tl.constexpr,
):
    """Stage 1: per-split partial attention with inline TQ dequant."""
    batch = tl.program_id(0)
    head = tl.program_id(1)
    split_id = tl.program_id(2)

    kv_head = head // kv_group_num
    seq_len = tl.load(Seq_lens + batch)

    # Load query vector
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + batch * stride_q_b + head * stride_q_h + offs_d).to(
        tl.float32
    )

    # Split range
    kv_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
    kv_start = kv_per_split * split_id
    kv_end = tl.minimum(kv_start + kv_per_split, seq_len)

    # Online softmax accumulators
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # QJL precompute
    if QJL_ENABLED:
        sign_byte_idx = offs_d // 8
        sign_bit_idx = (offs_d % 8).to(tl.int32)

    if kv_end > kv_start:
        for pos_start in range(kv_start, kv_end, BLOCK_KV):
            offs_pos = pos_start + tl.arange(0, BLOCK_KV)
            mask = offs_pos < kv_end

            # Block table → physical block + slot offset
            blk_idx = offs_pos // BLOCK_SIZE
            blk_off = offs_pos % BLOCK_SIZE
            phys_blk = tl.load(
                Block_table + batch * stride_bt_b + blk_idx * stride_bt_pos,
                mask=mask,
                other=0,
            ).to(tl.int64)

            # --- K dequant ---
            k_base = (
                phys_blk[:, None] * stride_kc_blk
                + blk_off[:, None] * stride_kc_slot
                + kv_head * stride_kc_head
                + offs_d[None, :] * stride_kc_dim
            )
            k_idx = tl.load(K_cache + k_base, mask=mask[:, None], other=0)
            k_vals = tl.load(Centroids + k_idx.to(tl.int32))

            kn_off = (
                phys_blk * stride_kn_blk
                + blk_off * stride_kn_slot
                + kv_head * stride_kn_head
            )
            k_nrm = tl.load(K_norms + kn_off, mask=mask, other=0.0)
            K = k_vals * k_nrm[:, None]

            if QJL_ENABLED:
                krs_off = (
                    phys_blk * stride_krs_blk
                    + blk_off * stride_krs_slot
                    + kv_head * stride_krs_head
                )
                k_rs = tl.load(K_res_scales + krs_off, mask=mask, other=0.0)
                k_sign_base = (
                    phys_blk[:, None] * stride_kc_blk
                    + blk_off[:, None] * stride_kc_slot
                    + kv_head * stride_kc_head
                    + (SIGN_DATA_OFFSET + sign_byte_idx)[None, :]
                    * stride_kc_dim
                )
                k_sign_bytes = tl.load(
                    K_cache + k_sign_base, mask=mask[:, None], other=0
                )
                k_sign_bits = (
                    k_sign_bytes.to(tl.int32) >> sign_bit_idx[None, :]
                ) & 1
                k_sign_vec = 2.0 * k_sign_bits.to(tl.float32) - 1.0
                K = K + k_sign_vec * (k_nrm * k_rs)[:, None]

            # Q · K score
            qk = tl.sum(q[None, :] * K, 1) * sm_scale
            qk = tl.where(mask, qk, float("-inf"))

            # --- V dequant ---
            v_base = (
                phys_blk[:, None] * stride_vc_blk
                + blk_off[:, None] * stride_vc_slot
                + kv_head * stride_vc_head
                + offs_d[None, :] * stride_vc_dim
            )
            v_idx = tl.load(V_cache + v_base, mask=mask[:, None], other=0)
            v_vals = tl.load(Centroids + v_idx.to(tl.int32))

            vn_off = (
                phys_blk * stride_vn_blk
                + blk_off * stride_vn_slot
                + kv_head * stride_vn_head
            )
            v_nrm = tl.load(V_norms + vn_off, mask=mask, other=0.0)
            V = v_vals * v_nrm[:, None]

            if QJL_ENABLED:
                vrs_off = (
                    phys_blk * stride_vrs_blk
                    + blk_off * stride_vrs_slot
                    + kv_head * stride_vrs_head
                )
                v_rs = tl.load(V_res_scales + vrs_off, mask=mask, other=0.0)
                v_sign_base = (
                    phys_blk[:, None] * stride_vc_blk
                    + blk_off[:, None] * stride_vc_slot
                    + kv_head * stride_vc_head
                    + (SIGN_DATA_OFFSET + sign_byte_idx)[None, :]
                    * stride_vc_dim
                )
                v_sign_bytes = tl.load(
                    V_cache + v_sign_base, mask=mask[:, None], other=0
                )
                v_sign_bits = (
                    v_sign_bytes.to(tl.int32) >> sign_bit_idx[None, :]
                ) & 1
                v_sign_vec = 2.0 * v_sign_bits.to(tl.float32) - 1.0
                V = V + v_sign_vec * (v_nrm * v_rs)[:, None]

            # Online softmax + accumulate
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc = acc * re_scale + tl.sum(p[:, None] * V, 0)
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

    # Store partial output + LSE
    out_base = (
        batch * stride_ao_b + head * stride_ao_h + split_id * stride_ao_s
    )
    tl.store(Att_Out + out_base + offs_d, acc / e_sum)
    tl.store(Att_Out + out_base + HEAD_DIM, e_max + tl.log(e_sum))


def tq_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    centroids: torch.Tensor,
    k_norms: torch.Tensor,
    v_norms: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    block_size: int,
    k_res_scales: torch.Tensor | None = None,
    v_res_scales: torch.Tensor | None = None,
    sign_data_offset: int = 0,
) -> None:
    """Fused TQ decode attention with split-KV reduction.

    Args:
        q: (batch, num_q_heads, head_dim) query
        key_cache, value_cache: (num_blocks, block_size, nkv, padded_dim) uint8
        output: (batch, num_q_heads, head_dim) pre-allocated output
        centroids: (n_levels,) float32 centroid values
        k_norms, v_norms: (num_blocks, block_size, nkv) float32 strided views
        block_table: (batch, max_blocks_per_seq) int32
        seq_lens: (batch,) int32
        k_res_scales, v_res_scales: QJL residual scales (optional)
        sign_data_offset: byte offset of sign bits in cache (head_dim + 4)
    """
    from vllm.v1.attention.ops.triton_decode_attention import (
        _fwd_kernel_stage2,
    )

    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = key_cache.shape[2]
    kv_group_num = num_q_heads // num_kv_heads
    qjl = k_res_scales is not None

    # Fixed split count — avoids GPU→CPU sync from seq_lens.max().item()
    num_kv_splits = 8

    BLOCK_KV = 16

    # Intermediate tensor: partial outputs + LSE
    att_out = torch.empty(
        (batch, num_q_heads, num_kv_splits, head_dim + 1),
        dtype=torch.float32,
        device=q.device,
    )

    # Dummy tensors for non-QJL params
    dummy_scales = k_norms  # unused but need valid ptr

    grid = (batch, num_q_heads, num_kv_splits)

    _tq_decode_stage1[grid](
        q,
        q.stride(0),
        q.stride(1),
        key_cache,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache,
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        centroids,
        k_norms,
        k_norms.stride(0),
        k_norms.stride(1),
        k_norms.stride(2),
        v_norms,
        v_norms.stride(0),
        v_norms.stride(1),
        v_norms.stride(2),
        k_res_scales if qjl else dummy_scales,
        k_res_scales.stride(0) if qjl else 0,
        k_res_scales.stride(1) if qjl else 0,
        k_res_scales.stride(2) if qjl else 0,
        v_res_scales if qjl else dummy_scales,
        v_res_scales.stride(0) if qjl else 0,
        v_res_scales.stride(1) if qjl else 0,
        v_res_scales.stride(2) if qjl else 0,
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        seq_lens,
        att_out,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        sm_scale,
        kv_group_num=kv_group_num,
        HEAD_DIM=head_dim,
        BLOCK_KV=BLOCK_KV,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=num_kv_splits,
        QJL_ENABLED=qjl,
        SIGN_DATA_OFFSET=sign_data_offset if qjl else 0,
        num_warps=2,
        num_stages=2,
    )

    # Stage 2: merge partial outputs across splits
    lse = torch.empty(
        (batch, num_q_heads), dtype=torch.float32, device=q.device
    )

    grid2 = (batch, num_q_heads)
    BLOCK_DV = triton.next_power_of_2(head_dim)

    _fwd_kernel_stage2[grid2](
        att_out,
        output,
        lse,
        seq_lens,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=head_dim,
    )
