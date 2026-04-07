# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end attention correctness tests for TurboQuant backend.

Verifies that the full TurboQuant pipeline (encode -> fused attention -> decode)
produces outputs close to bf16 SDPA reference, within quantization tolerance.

The test cannot use the standard test_attention_backends.py infrastructure
because:
1. TurboQuant has a different KV cache shape (uint8, padded dimensions)
2. TQ requires query rotation and output inverse-rotation
3. The comparison must use cosine similarity (not exact match) due to
   quantization
4. The TQ encode path is separate from the standard reshape_and_cache

Each test:
1. Creates random Q, K, V in bf16
2. Computes SDPA reference on the raw bf16 data
3. Creates a TQ KV cache, encodes K/V via do_kv_cache_update()
4. Runs TQ attention via forward()
5. Compares output vs reference using cosine similarity
"""

import math

import pytest
import torch

from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
)
from vllm.v1.attention.backends.turboquant_config import parse_tq_preset

# Cosine similarity thresholds per preset.
# These are intentionally conservative; real-model quality is higher
# because activations are structured (not random Gaussian).
_THRESHOLD = {
    "tq-k8v8": 0.98,
    "tq-k8fv4": 0.99,
    "tq-k4v4": 0.90,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compute causal attention using PyTorch SDPA as ground truth.

    Args:
        query: (num_tokens, num_q_heads, head_dim)
        key: (seq_len, num_kv_heads, head_dim)
        value: (seq_len, num_kv_heads, head_dim)
        scale: attention scale factor

    Returns:
        output: (num_tokens, num_q_heads, head_dim)
    """
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    num_queries_per_kv = num_q_heads // num_kv_heads

    if num_queries_per_kv > 1:
        key = key.repeat_interleave(num_queries_per_kv, dim=1)
        value = value.repeat_interleave(num_queries_per_kv, dim=1)

    # Transpose to (batch=1, heads, seq, dim) for SDPA
    q = query.unsqueeze(0).transpose(1, 2)  # (1, H, T_q, D)
    k = key.unsqueeze(0).transpose(1, 2)  # (1, H, T_k, D)
    v = value.unsqueeze(0).transpose(1, 2)  # (1, H, T_k, D)

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        scale=scale,
    )
    return out.transpose(1, 2).squeeze(0)  # (T_q, H, D)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors along last dim."""
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos.mean().item()


def _run_tq_attention_test(
    preset_str: str,
    num_tokens: int,
    context_len: int,
    num_q_heads: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    block_size: int = 16,
    q: torch.Tensor | None = None,
    k_all: torch.Tensor | None = None,
    v_all: torch.Tensor | None = None,
) -> float:
    """Run a single TQ attention test and return cosine similarity.

    Creates random data (or uses provided tensors), encodes into a TQ
    KV cache, runs fused attention, and compares against bf16 SDPA.

    Args:
        preset_str: TQ preset name (e.g. "tq-k8v8").
        num_tokens: Number of new query tokens (decode step).
        context_len: Number of previously-cached context tokens.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of KV heads (must divide num_q_heads).
        head_dim: Head dimension (power of 2).
        block_size: KV cache block size.
        q: Optional pre-created query tensor.
        k_all: Optional pre-created key tensor (context + new).
        v_all: Optional pre-created value tensor (context + new).

    Returns:
        Mean cosine similarity between TQ output and SDPA reference.
    """
    device = torch.device("cuda:0")
    scale = 1.0 / math.sqrt(head_dim)
    seq_len = context_len + num_tokens

    # 1. Generate data
    if q is None:
        q = torch.randn(
            num_tokens,
            num_q_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
    if k_all is None:
        k_all = torch.randn(
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
    if v_all is None:
        v_all = torch.randn(
            seq_len,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

    k_context = k_all[:context_len]
    v_context = v_all[:context_len]
    k_new = k_all[context_len:]
    v_new = v_all[context_len:]

    # 2. SDPA reference (uses ALL K/V, causal)
    ref_out = _sdpa_reference(q, k_all, v_all, scale)

    # 3. Create TQ backend and KV cache
    preset = parse_tq_preset(preset_str)
    padded_dim = preset.padded_cache_dim(head_dim)
    num_blocks = (seq_len + block_size - 1) // block_size + 1  # +1 safety

    kv_cache = torch.zeros(
        num_blocks,
        2,
        block_size,
        num_kv_heads,
        padded_dim,
        dtype=torch.uint8,
        device=device,
    )

    impl = TurboQuantAttentionImpl(
        num_heads=num_q_heads,
        head_size=head_dim,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=preset_str,
    )

    # 4. Encode context K/V into cache (simulate prefill)
    if context_len > 0:
        context_slot_mapping = torch.arange(
            context_len, dtype=torch.int64, device=device
        )
        impl.do_kv_cache_update(
            None, k_context, v_context, kv_cache, context_slot_mapping
        )

    # 5. Encode new K/V (simulate decode step)
    new_slot_mapping = torch.arange(
        context_len,
        context_len + num_tokens,
        dtype=torch.int64,
        device=device,
    )
    impl.do_kv_cache_update(None, k_new, v_new, kv_cache, new_slot_mapping)

    # 6. Build metadata (single-sequence decode)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(
        max_blocks_per_seq, dtype=torch.int32, device=device
    ).unsqueeze(0)

    metadata = TurboQuantMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        max_seq_len=seq_len,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=new_slot_mapping,
    )

    # 7. Run TQ attention
    output = torch.empty(
        num_tokens,
        num_q_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    tq_out = impl.forward(None, q, k_new, v_new, kv_cache, metadata, output)

    # 8. Compare
    cos_sim = _cosine_similarity(tq_out[:num_tokens], ref_out)
    return cos_sim


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTurboQuantAttentionCorrectness:
    """End-to-end correctness tests for TurboQuant attention."""

    # -- Basic decode correctness --

    @pytest.mark.parametrize("preset", ["tq-k8v8", "tq-k4v4", "tq-k8fv4"])
    @pytest.mark.parametrize("num_tokens", [1, 4, 16])
    @pytest.mark.parametrize("seq_len", [32, 128, 512])
    def test_decode_correctness(self, preset: str, num_tokens: int, seq_len: int):
        """Test TQ attention vs SDPA for decode (short query, long context).

        The query has ``num_tokens`` new tokens attending over ``seq_len``
        context tokens already in the KV cache. Cosine similarity must
        exceed the preset-specific threshold.
        """
        cos_sim = _run_tq_attention_test(
            preset_str=preset,
            num_tokens=num_tokens,
            context_len=seq_len,
        )
        threshold = _THRESHOLD[preset]
        assert cos_sim > threshold, (
            f"Cosine similarity {cos_sim:.4f} below threshold "
            f"{threshold} for preset={preset}, "
            f"num_tokens={num_tokens}, seq_len={seq_len}"
        )

    # -- GQA --

    @pytest.mark.parametrize("preset", ["tq-k8v8", "tq-k4v4"])
    def test_gqa(self, preset: str):
        """GQA: num_q_heads=32, num_kv_heads=8 (4x ratio)."""
        cos_sim = _run_tq_attention_test(
            preset_str=preset,
            num_tokens=4,
            context_len=128,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )
        threshold = _THRESHOLD[preset]
        assert cos_sim > threshold, (
            f"GQA cosine similarity {cos_sim:.4f} below threshold "
            f"{threshold} for preset={preset}"
        )

    # -- Head dim variants --

    @pytest.mark.parametrize("preset", ["tq-k8v8"])
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_head_dim_variants(self, preset: str, head_dim: int):
        """Different head dimensions (all power-of-2)."""
        cos_sim = _run_tq_attention_test(
            preset_str=preset,
            num_tokens=4,
            context_len=64,
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=head_dim,
        )
        threshold = _THRESHOLD[preset]
        assert cos_sim > threshold, (
            f"Head dim {head_dim} cosine similarity {cos_sim:.4f} "
            f"below threshold {threshold} for preset={preset}"
        )

    # -- FP8 key mode --

    def test_fp8_key_no_q_rotation(self):
        """Verify FP8 key mode produces high-quality results.

        FP8 keys are near-lossless so the cosine similarity should be
        very high (> 0.99). This also implicitly verifies that Q is NOT
        rotated in FP8 key mode (K is in original space).
        """
        cos_sim = _run_tq_attention_test(
            preset_str="tq-k8fv4",
            num_tokens=4,
            context_len=128,
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=128,
        )
        threshold = _THRESHOLD["tq-k8fv4"]
        assert cos_sim > threshold, (
            f"FP8 key cosine similarity {cos_sim:.4f} below threshold {threshold}"
        )

    def test_fp8_key_gqa(self):
        """FP8 key mode with GQA (32 q heads, 8 kv heads)."""
        cos_sim = _run_tq_attention_test(
            preset_str="tq-k8fv4",
            num_tokens=4,
            context_len=128,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
        )
        threshold = _THRESHOLD["tq-k8fv4"]
        assert cos_sim > threshold, (
            f"FP8 key GQA cosine similarity {cos_sim:.4f} below threshold {threshold}"
        )

    # -- Numerical stability --

    @pytest.mark.parametrize("preset", ["tq-k8v8", "tq-k4v4"])
    def test_numerical_stability_large_norms(self, preset: str):
        """Test with large-norm vectors (scaled up by 100x).

        TQ normalizes vectors before quantizing, so large norms should
        not affect quality — the norm is stored separately.
        """
        device = torch.device("cuda:0")
        num_tokens = 4
        context_len = 64
        num_q_heads = 8
        num_kv_heads = 4
        head_dim = 128
        seq_len = context_len + num_tokens

        q = (
            torch.randn(
                num_tokens,
                num_q_heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 100.0
        )
        k_all = (
            torch.randn(
                seq_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 100.0
        )
        v_all = (
            torch.randn(
                seq_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 100.0
        )

        cos_sim = _run_tq_attention_test(
            preset_str=preset,
            num_tokens=num_tokens,
            context_len=context_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q=q,
            k_all=k_all,
            v_all=v_all,
        )
        threshold = _THRESHOLD[preset]
        assert cos_sim > threshold, (
            f"Large-norm cosine similarity {cos_sim:.4f} below "
            f"threshold {threshold} for preset={preset}"
        )

    @pytest.mark.parametrize("preset", ["tq-k8v8", "tq-k4v4"])
    def test_numerical_stability_small_norms(self, preset: str):
        """Test with near-zero vectors (scaled down by 1e-3).

        TQ should handle small-norm vectors gracefully due to the
        epsilon in the normalization (1e-10 floor).
        """
        device = torch.device("cuda:0")
        num_tokens = 4
        context_len = 64
        num_q_heads = 8
        num_kv_heads = 4
        head_dim = 128
        seq_len = context_len + num_tokens

        q = (
            torch.randn(
                num_tokens,
                num_q_heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 1e-3
        )
        k_all = (
            torch.randn(
                seq_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 1e-3
        )
        v_all = (
            torch.randn(
                seq_len,
                num_kv_heads,
                head_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            * 1e-3
        )

        cos_sim = _run_tq_attention_test(
            preset_str=preset,
            num_tokens=num_tokens,
            context_len=context_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q=q,
            k_all=k_all,
            v_all=v_all,
        )
        # Small-norm vectors may have slightly lower similarity due to
        # bf16 precision limits at small magnitudes; use a relaxed threshold.
        relaxed = max(_THRESHOLD[preset] - 0.05, 0.85)
        assert cos_sim > relaxed, (
            f"Small-norm cosine similarity {cos_sim:.4f} below "
            f"relaxed threshold {relaxed} for preset={preset}"
        )

    # -- Single-token decode (typical autoregressive) --

    @pytest.mark.parametrize("preset", ["tq-k8v8", "tq-k8fv4"])
    def test_single_token_decode(self, preset: str):
        """Single-token decode: the most common inference scenario.

        1 new token attending over a long context (256 tokens).
        """
        cos_sim = _run_tq_attention_test(
            preset_str=preset,
            num_tokens=1,
            context_len=256,
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=128,
        )
        threshold = _THRESHOLD[preset]
        assert cos_sim > threshold, (
            f"Single-token decode cosine similarity {cos_sim:.4f} "
            f"below threshold {threshold} for preset={preset}"
        )

    # -- Prefill-only (all tokens are new, no context) --

    @pytest.mark.parametrize("preset", ["tq-k8v8", "tq-k4v4"])
    def test_prefill_only(self, preset: str):
        """Prefill-only: context_len=0, all tokens are new.

        Verifies that the encode + attention path works when there is
        no pre-existing context in the KV cache.
        """
        cos_sim = _run_tq_attention_test(
            preset_str=preset,
            num_tokens=32,
            context_len=0,
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=128,
        )
        threshold = _THRESHOLD[preset]
        assert cos_sim > threshold, (
            f"Prefill-only cosine similarity {cos_sim:.4f} below "
            f"threshold {threshold} for preset={preset}"
        )

    # -- MHA (num_q_heads == num_kv_heads) --

    @pytest.mark.parametrize("preset", ["tq-k8v8"])
    def test_mha(self, preset: str):
        """MHA: num_q_heads == num_kv_heads (no GQA)."""
        cos_sim = _run_tq_attention_test(
            preset_str=preset,
            num_tokens=4,
            context_len=64,
            num_q_heads=8,
            num_kv_heads=8,
            head_dim=128,
        )
        threshold = _THRESHOLD[preset]
        assert cos_sim > threshold, (
            f"MHA cosine similarity {cos_sim:.4f} below threshold "
            f"{threshold} for preset={preset}"
        )

    # -- Different block sizes --

    @pytest.mark.parametrize("block_size", [16, 32, 64])
    def test_block_sizes(self, block_size: int):
        """Different KV cache block sizes."""
        cos_sim = _run_tq_attention_test(
            preset_str="tq-k8v8",
            num_tokens=4,
            context_len=128,
            num_q_heads=8,
            num_kv_heads=4,
            head_dim=128,
            block_size=block_size,
        )
        threshold = _THRESHOLD["tq-k8v8"]
        assert cos_sim > threshold, (
            f"Block size {block_size} cosine similarity {cos_sim:.4f} "
            f"below threshold {threshold}"
        )
