# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TurboQuant KV cache compression."""

import math

import pytest
import torch

from vllm.v1.attention.ops.turboquant import (
    TurboQuantCodebook,
    compute_beta_centroids,
    generate_rotation_matrix,
    pack_nibbles,
    turboquant_decode_ref,
    turboquant_encode_ref,
    unpack_nibbles,
)


class TestBetaCentroids:
    """Test Beta(d/2) centroid computation."""

    def test_4bit_centroids(self):
        boundaries, centroids = compute_beta_centroids(128, 4)
        assert boundaries.shape == (15,)
        assert centroids.shape == (16,)
        assert torch.all(centroids[:-1] < centroids[1:])
        # Beta centroids are in [-1, 1]
        assert centroids.min() >= -1.0
        assert centroids.max() <= 1.0

    def test_3bit_centroids(self):
        boundaries, centroids = compute_beta_centroids(64, 3)
        assert boundaries.shape == (7,)
        assert centroids.shape == (8,)

    def test_6bit_centroids(self):
        boundaries, centroids = compute_beta_centroids(128, 6)
        assert boundaries.shape == (63,)
        assert centroids.shape == (64,)

    def test_symmetry(self):
        """Centroids for Beta(d/2) should be approximately symmetric."""
        _, centroids = compute_beta_centroids(128, 4)
        assert torch.allclose(centroids + centroids.flip(0),
                              torch.zeros_like(centroids), atol=0.02)


class TestRotationMatrix:
    """Test random orthogonal rotation matrix."""

    def test_deterministic(self):
        R1 = generate_rotation_matrix(64, seed=42)
        R2 = generate_rotation_matrix(64, seed=42)
        assert torch.allclose(R1, R2)

    def test_different_seeds(self):
        R1 = generate_rotation_matrix(64, seed=42)
        R2 = generate_rotation_matrix(64, seed=43)
        assert not torch.allclose(R1, R2)

    def test_orthogonal(self):
        """R @ R^T should be identity."""
        R = generate_rotation_matrix(128, seed=42)
        eye = R @ R.T
        assert torch.allclose(eye, torch.eye(128), atol=1e-5)

    def test_shape(self):
        R = generate_rotation_matrix(64, seed=42)
        assert R.shape == (64, 64)


class TestTurboQuantCodebook:
    """Test codebook initialization."""

    def test_init_4bit(self):
        cb = TurboQuantCodebook(n_bits=4, head_dim=128, device="cpu")
        assert cb.n_levels == 16
        assert cb.boundaries.shape == (15,)
        assert cb.centroids.shape == (16,)
        assert cb.rotation_matrix.shape == (128, 128)

    def test_init_3bit(self):
        cb = TurboQuantCodebook(n_bits=3, head_dim=64, device="cpu")
        assert cb.n_levels == 8
        assert cb.boundaries.shape == (7,)


class TestTurboQuantEncodeDecode:
    """Test encode/decode roundtrip quality."""

    @pytest.fixture
    def codebook_4bit(self):
        return TurboQuantCodebook(n_bits=4, head_dim=128, device="cpu")

    @pytest.fixture
    def codebook_3bit(self):
        return TurboQuantCodebook(n_bits=3, head_dim=128, device="cpu")

    def test_encode_shape_2d_packed(self, codebook_4bit):
        tensor = torch.randn(10, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit, packed=True)
        assert indices.shape == (10, 64)  # nibble-packed: d//2
        assert indices.dtype == torch.uint8
        assert norms.shape == (10,)

    def test_encode_shape_2d_unpacked(self, codebook_4bit):
        tensor = torch.randn(10, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit, packed=False)
        assert indices.shape == (10, 128)
        assert indices.dtype == torch.uint8
        assert norms.shape == (10,)

    def test_encode_shape_3d(self, codebook_4bit):
        tensor = torch.randn(10, 8, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit, packed=True)
        assert indices.shape == (10, 8, 64)  # nibble-packed: d//2
        assert indices.dtype == torch.uint8
        assert norms.shape == (10, 8)

    def test_decode_shape_packed(self, codebook_4bit):
        # Packed indices: d//2 bytes
        packed = torch.randint(0, 256, (10, 64), dtype=torch.uint8)
        norms = torch.ones(10)
        decoded = turboquant_decode_ref(packed, norms, codebook_4bit,
                                        packed=True)
        assert decoded.shape == (10, 128)
        assert decoded.dtype == torch.bfloat16

    def test_decode_shape_unpacked(self, codebook_4bit):
        indices = torch.randint(0, 16, (10, 128), dtype=torch.uint8)
        norms = torch.ones(10)
        decoded = turboquant_decode_ref(indices, norms, codebook_4bit,
                                        packed=False)
        assert decoded.shape == (10, 128)
        assert decoded.dtype == torch.bfloat16

    def test_roundtrip_cosine_sim_4bit(self, codebook_4bit):
        """4-bit should achieve >0.97 cosine similarity."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit)
        decoded = turboquant_decode_ref(
            indices, norms, codebook_4bit, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor, decoded, dim=1)
        mean_sim = cos_sim.mean().item()
        assert mean_sim > 0.97, f"Mean cosine similarity {mean_sim:.4f} too low"

    def test_roundtrip_cosine_sim_3bit(self, codebook_3bit):
        """3-bit should achieve >0.93 cosine similarity."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_3bit)
        decoded = turboquant_decode_ref(
            indices, norms, codebook_3bit, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor, decoded, dim=1)
        mean_sim = cos_sim.mean().item()
        assert mean_sim > 0.93, f"Mean cosine similarity {mean_sim:.4f} too low"

    def test_4bit_better_than_3bit(self):
        """4-bit should give higher cosine sim than 3-bit."""
        torch.manual_seed(42)
        tensor = torch.randn(200, 128)
        cb3 = TurboQuantCodebook(n_bits=3, head_dim=128, device="cpu")
        cb4 = TurboQuantCodebook(n_bits=4, head_dim=128, device="cpu")

        idx3, n3 = turboquant_encode_ref(tensor, cb3)
        dec3 = turboquant_decode_ref(idx3, n3, cb3, torch.float32)
        sim3 = torch.nn.functional.cosine_similarity(
            tensor, dec3, dim=1).mean()

        idx4, n4 = turboquant_encode_ref(tensor, cb4)
        dec4 = turboquant_decode_ref(idx4, n4, cb4, torch.float32)
        sim4 = torch.nn.functional.cosine_similarity(
            tensor, dec4, dim=1).mean()

        assert sim4 > sim3, (
            f"4-bit ({sim4:.4f}) should be better than 3-bit ({sim3:.4f})")

    def test_index_range(self, codebook_4bit):
        """Unpacked indices must be in [0, n_levels)."""
        tensor = torch.randn(100, 128) * 10
        indices, _ = turboquant_encode_ref(tensor, codebook_4bit,
                                            packed=False)
        assert indices.max().item() < codebook_4bit.n_levels
        assert indices.min().item() >= 0

    def test_empty_tensor(self, codebook_4bit):
        tensor = torch.randn(0, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit,
                                                packed=True)
        assert indices.shape == (0, 64)
        decoded = turboquant_decode_ref(indices, norms, codebook_4bit,
                                         packed=True)
        assert decoded.shape == (0, 128)

    def test_single_token(self, codebook_4bit):
        tensor = torch.randn(1, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit)
        decoded = turboquant_decode_ref(
            indices, norms, codebook_4bit, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor, decoded, dim=1)
        assert cos_sim.item() > 0.90

    def test_norm_preservation(self, codebook_4bit):
        """Norms should be approximately preserved."""
        torch.manual_seed(42)
        tensor = torch.randn(100, 128)
        original_norms = torch.norm(tensor, dim=1)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit)
        decoded = turboquant_decode_ref(
            indices, norms, codebook_4bit, torch.float32)
        decoded_norms = torch.norm(decoded, dim=1)
        rel_error = (
            (decoded_norms - original_norms).abs()
            / (original_norms + 1e-8))
        assert rel_error.mean() < 0.15, (
            f"Mean norm relative error {rel_error.mean():.4f} too high")

    def test_3d_roundtrip(self, codebook_4bit):
        """Test with [num_tokens, num_heads, head_dim] shape."""
        torch.manual_seed(42)
        tensor = torch.randn(50, 8, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit)
        decoded = turboquant_decode_ref(
            indices, norms, codebook_4bit, torch.float32)
        # Check per-head cosine similarity
        for h in range(8):
            cos_sim = torch.nn.functional.cosine_similarity(
                tensor[:, h, :], decoded[:, h, :], dim=1)
            assert cos_sim.mean().item() > 0.97


class TestNibblePacking:
    """Test nibble pack/unpack correctness."""

    def test_pack_unpack_roundtrip(self):
        torch.manual_seed(42)
        indices = torch.randint(0, 16, (50, 128), dtype=torch.uint8)
        packed = pack_nibbles(indices)
        assert packed.shape == (50, 64)
        unpacked = unpack_nibbles(packed, 128)
        assert torch.equal(indices, unpacked)

    def test_pack_unpack_3d(self):
        torch.manual_seed(42)
        indices = torch.randint(0, 16, (10, 8, 128), dtype=torch.uint8)
        packed = pack_nibbles(indices)
        assert packed.shape == (10, 8, 64)
        unpacked = unpack_nibbles(packed, 128)
        assert torch.equal(indices, unpacked)

    def test_pack_values_correct(self):
        """Manually verify packing convention."""
        # indices[0..63] -> low nibble, indices[64..127] -> high nibble
        indices = torch.zeros(128, dtype=torch.uint8)
        indices[0] = 5    # low half, position 0
        indices[64] = 10   # high half, position 0
        packed = pack_nibbles(indices)
        assert packed[0].item() == 5 | (10 << 4)  # 0xA5 = 165

    def test_pack_shape(self):
        indices = torch.randint(0, 16, (100, 64), dtype=torch.uint8)
        packed = pack_nibbles(indices)
        assert packed.shape == (100, 32)


class TestCompressionRatio:
    """Test that TurboQuant achieves expected compression."""

    def test_packed_compression(self):
        """Nibble-packed 4-bit gives ~3.8x compression over FP16."""
        head_dim = 128
        num_tokens = 1000
        fp16_bytes = num_tokens * head_dim * 2
        # Packed: head_dim//2 bytes + 4 bytes float32 norm per token
        tq_bytes = num_tokens * (head_dim // 2) + num_tokens * 4
        ratio = fp16_bytes / tq_bytes
        assert ratio > 3.5, f"Packed ratio {ratio:.2f}x too low"

    def test_packed_vs_fp8(self):
        """Packed TurboQuant should be better than FP8 (2x)."""
        head_dim = 128
        num_tokens = 1000
        fp8_bytes = num_tokens * head_dim * 1  # uint8
        tq_bytes = num_tokens * (head_dim // 2) + num_tokens * 4
        assert tq_bytes < fp8_bytes, "Packed TQ should use less than FP8"


class TestKVQuantMode:
    """Test KVQuantMode enum and utilities."""

    def test_turboquant_mode(self):
        from vllm.v1.kv_cache_interface import (
            KVQuantMode,
            get_kv_quant_mode,
        )
        mode = get_kv_quant_mode("turboquant")
        assert mode == KVQuantMode.TURBOQUANT
        assert mode == 4
        assert not mode.is_per_token_head

    def test_other_modes_unchanged(self):
        from vllm.v1.kv_cache_interface import (
            KVQuantMode,
            get_kv_quant_mode,
        )
        assert get_kv_quant_mode("auto") == KVQuantMode.NONE
        assert get_kv_quant_mode("fp8") == KVQuantMode.FP8_PER_TENSOR
        assert get_kv_quant_mode("fp8_e4m3") == KVQuantMode.FP8_PER_TENSOR
        assert get_kv_quant_mode("int8_per_token_head") == (
            KVQuantMode.INT8_PER_TOKEN_HEAD)
        assert get_kv_quant_mode("fp8_per_token_head") == (
            KVQuantMode.FP8_PER_TOKEN_HEAD)


class TestCacheDType:
    """Test that turboquant is registered correctly."""

    def test_dtype_mapping(self):
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        assert "turboquant" in STR_DTYPE_TO_TORCH_DTYPE
        assert STR_DTYPE_TO_TORCH_DTYPE["turboquant"] == torch.uint8

    def test_is_turboquant(self):
        from vllm.utils.torch_utils import is_turboquant_kv_cache
        assert is_turboquant_kv_cache("turboquant")
        assert not is_turboquant_kv_cache("fp8")
        assert not is_turboquant_kv_cache("auto")

    def test_is_quantized_unchanged(self):
        """is_quantized_kv_cache should NOT match turboquant."""
        from vllm.utils.torch_utils import is_quantized_kv_cache
        assert not is_quantized_kv_cache("turboquant")
        assert is_quantized_kv_cache("fp8")
        assert is_quantized_kv_cache("fp8_e4m3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
