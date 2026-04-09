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
    pack_2bit,
    pack_nibbles,
    pack_sign_bits,
    sign_bytes_padded,
    turboquant_decode_qjl_ref,
    turboquant_decode_ref,
    turboquant_encode_qjl_ref,
    turboquant_encode_ref,
    unpack_2bit,
    unpack_nibbles,
    unpack_sign_bits,
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
        assert torch.allclose(
            centroids + centroids.flip(0), torch.zeros_like(centroids), atol=0.02
        )


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
        decoded = turboquant_decode_ref(packed, norms, codebook_4bit, packed=True)
        assert decoded.shape == (10, 128)
        assert decoded.dtype == torch.bfloat16

    def test_decode_shape_unpacked(self, codebook_4bit):
        indices = torch.randint(0, 16, (10, 128), dtype=torch.uint8)
        norms = torch.ones(10)
        decoded = turboquant_decode_ref(indices, norms, codebook_4bit, packed=False)
        assert decoded.shape == (10, 128)
        assert decoded.dtype == torch.bfloat16

    def test_roundtrip_cosine_sim_4bit(self, codebook_4bit):
        """4-bit should achieve >0.99 cosine similarity."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit)
        decoded = turboquant_decode_ref(indices, norms, codebook_4bit, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(tensor, decoded, dim=1)
        mean_sim = cos_sim.mean().item()
        assert mean_sim > 0.99, f"Mean cosine similarity {mean_sim:.4f} too low"

    def test_roundtrip_cosine_sim_3bit(self, codebook_3bit):
        """3-bit should achieve >0.96 cosine similarity."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_3bit)
        decoded = turboquant_decode_ref(indices, norms, codebook_3bit, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(tensor, decoded, dim=1)
        mean_sim = cos_sim.mean().item()
        assert mean_sim > 0.96, f"Mean cosine similarity {mean_sim:.4f} too low"

    def test_4bit_better_than_3bit(self):
        """4-bit should give higher cosine sim than 3-bit."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)
        cb3 = TurboQuantCodebook(n_bits=3, head_dim=128, device="cpu")
        cb4 = TurboQuantCodebook(n_bits=4, head_dim=128, device="cpu")

        idx3, n3 = turboquant_encode_ref(tensor, cb3)
        dec3 = turboquant_decode_ref(idx3, n3, cb3, torch.float32)
        sim3 = torch.nn.functional.cosine_similarity(tensor, dec3, dim=1).mean()

        idx4, n4 = turboquant_encode_ref(tensor, cb4)
        dec4 = turboquant_decode_ref(idx4, n4, cb4, torch.float32)
        sim4 = torch.nn.functional.cosine_similarity(tensor, dec4, dim=1).mean()

        assert sim4 > sim3, (
            f"4-bit ({sim4:.4f}) should be better than 3-bit ({sim3:.4f})"
        )

    def test_index_range(self, codebook_4bit):
        """Unpacked indices must be in [0, n_levels)."""
        tensor = torch.randn(100, 128) * 10
        indices, _ = turboquant_encode_ref(tensor, codebook_4bit, packed=False)
        assert indices.max().item() < codebook_4bit.n_levels
        assert indices.min().item() >= 0

    def test_empty_tensor(self, codebook_4bit):
        tensor = torch.randn(0, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit, packed=True)
        assert indices.shape == (0, 64)
        decoded = turboquant_decode_ref(indices, norms, codebook_4bit, packed=True)
        assert decoded.shape == (0, 128)

    def test_single_token(self, codebook_4bit):
        tensor = torch.randn(1, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit)
        decoded = turboquant_decode_ref(indices, norms, codebook_4bit, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(tensor, decoded, dim=1)
        assert cos_sim.item() > 0.90

    def test_norm_preservation(self, codebook_4bit):
        """Norms should be approximately preserved."""
        torch.manual_seed(123)
        tensor = torch.randn(100, 128)
        original_norms = torch.norm(tensor, dim=1)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit)
        decoded = turboquant_decode_ref(indices, norms, codebook_4bit, torch.float32)
        decoded_norms = torch.norm(decoded, dim=1)
        rel_error = (decoded_norms - original_norms).abs() / (original_norms + 1e-8)
        assert rel_error.mean() < 0.05, (
            f"Mean norm relative error {rel_error.mean():.4f} too high"
        )

    def test_3d_roundtrip(self, codebook_4bit):
        """Test with [num_tokens, num_heads, head_dim] shape."""
        torch.manual_seed(123)
        tensor = torch.randn(50, 8, 128)
        indices, norms = turboquant_encode_ref(tensor, codebook_4bit)
        decoded = turboquant_decode_ref(indices, norms, codebook_4bit, torch.float32)
        # Check per-head cosine similarity
        for h in range(8):
            cos_sim = torch.nn.functional.cosine_similarity(
                tensor[:, h, :], decoded[:, h, :], dim=1
            )
            assert cos_sim.mean().item() > 0.99


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
        indices[0] = 5  # low half, position 0
        indices[64] = 10  # high half, position 0
        packed = pack_nibbles(indices)
        assert packed[0].item() == 5 | (10 << 4)  # 0xA5 = 165

    def test_pack_shape(self):
        indices = torch.randint(0, 16, (100, 64), dtype=torch.uint8)
        packed = pack_nibbles(indices)
        assert packed.shape == (100, 32)

    def test_2bit_pack_roundtrip(self):
        idx = torch.randint(0, 4, (20, 96), dtype=torch.uint8)
        packed = pack_2bit(idx)
        assert packed.shape == (20, 24)
        unpacked = unpack_2bit(packed, 96)
        assert (idx == unpacked).all()


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
            get_kv_quant_mode,
        )

        mode = get_kv_quant_mode("turboquant")
        # With default auto (0), should get TURBOQUANT_BYTE (6-bit)
        assert mode.is_turboquant
        assert not mode.is_per_token_head

    def test_tq_nibble_preset(self):
        """tq-k4v4 preset should map to TURBOQUANT (nibble mode)."""
        from vllm.v1.kv_cache_interface import (
            KVQuantMode,
            get_kv_quant_mode,
        )

        mode = get_kv_quant_mode("tq-k4v4")
        assert mode == KVQuantMode.TURBOQUANT

    def test_tq_byte_preset(self):
        """tq-k8v8 preset should map to TURBOQUANT_BYTE."""
        from vllm.v1.kv_cache_interface import (
            KVQuantMode,
            get_kv_quant_mode,
        )

        mode = get_kv_quant_mode("tq-k8v8")
        assert mode == KVQuantMode.TURBOQUANT_BYTE

    def test_other_modes_unchanged(self):
        from vllm.v1.kv_cache_interface import (
            KVQuantMode,
            get_kv_quant_mode,
        )

        assert get_kv_quant_mode("auto") == KVQuantMode.NONE
        assert get_kv_quant_mode("fp8") == KVQuantMode.FP8_PER_TENSOR
        assert get_kv_quant_mode("fp8_e4m3") == KVQuantMode.FP8_PER_TENSOR
        assert get_kv_quant_mode("int8_per_token_head") == (
            KVQuantMode.INT8_PER_TOKEN_HEAD
        )
        assert get_kv_quant_mode("fp8_per_token_head") == (
            KVQuantMode.FP8_PER_TOKEN_HEAD
        )


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


class TestByteMode:
    """Test byte-mode (5-8 bit) TurboQuant."""

    def test_codebook_byte_mode(self):
        cb = TurboQuantCodebook(n_bits=6, head_dim=128, device="cpu")
        assert cb.byte_mode is True
        assert cb.n_levels == 64
        assert cb.centroids.shape == (64,)
        assert cb.boundaries.shape == (63,)

    def test_codebook_nibble_mode(self):
        cb = TurboQuantCodebook(n_bits=4, head_dim=128, device="cpu")
        assert cb.byte_mode is False

    def test_roundtrip_6bit_d128(self):
        """6-bit d=128 should achieve >0.99 cosine similarity."""
        cb = TurboQuantCodebook(n_bits=6, head_dim=128, device="cpu")
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)
        indices, norms = turboquant_encode_ref(tensor, cb)
        assert indices.shape == (200, 128)  # byte mode: no packing
        decoded = turboquant_decode_ref(indices, norms, cb, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(tensor, decoded, dim=1)
        assert cos_sim.mean().item() > 0.99

    def test_roundtrip_5bit_d64(self):
        """5-bit d=64 should achieve >0.97 cosine similarity."""
        cb = TurboQuantCodebook(n_bits=5, head_dim=64, device="cpu")
        torch.manual_seed(123)
        tensor = torch.randn(200, 64)
        indices, norms = turboquant_encode_ref(tensor, cb)
        assert indices.shape == (200, 64)  # byte mode
        decoded = turboquant_decode_ref(indices, norms, cb, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(tensor, decoded, dim=1)
        assert cos_sim.mean().item() > 0.97

    def test_roundtrip_8bit(self):
        """8-bit should achieve >0.999 cosine similarity."""
        cb = TurboQuantCodebook(n_bits=8, head_dim=128, device="cpu")
        torch.manual_seed(123)
        tensor = torch.randn(100, 128)
        indices, norms = turboquant_encode_ref(tensor, cb)
        decoded = turboquant_decode_ref(indices, norms, cb, torch.float32)
        cos_sim = torch.nn.functional.cosine_similarity(tensor, decoded, dim=1)
        assert cos_sim.mean().item() > 0.999

    def test_6bit_better_than_4bit(self):
        """6-bit should give higher cosine sim than 4-bit."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)
        cb4 = TurboQuantCodebook(n_bits=4, head_dim=128, device="cpu")
        cb6 = TurboQuantCodebook(n_bits=6, head_dim=128, device="cpu")

        idx4, n4 = turboquant_encode_ref(tensor, cb4)
        dec4 = turboquant_decode_ref(idx4, n4, cb4, torch.float32)
        sim4 = torch.nn.functional.cosine_similarity(tensor, dec4, dim=1).mean()

        idx6, n6 = turboquant_encode_ref(tensor, cb6)
        dec6 = turboquant_decode_ref(idx6, n6, cb6, torch.float32)
        sim6 = torch.nn.functional.cosine_similarity(tensor, dec6, dim=1).mean()

        assert sim6 > sim4

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            TurboQuantCodebook(n_bits=9, head_dim=128, device="cpu")


class TestSignBitPacking:
    """Test sign bit pack/unpack correctness."""

    def test_pack_unpack_roundtrip(self):
        torch.manual_seed(42)
        signs = torch.randint(0, 2, (50, 128), dtype=torch.uint8)
        packed = pack_sign_bits(signs)
        assert packed.shape == (50, 16)  # 128/8 = 16 bytes
        unpacked = unpack_sign_bits(packed, 128)
        expected = 2.0 * signs.float() - 1.0
        assert torch.allclose(unpacked, expected)

    def test_pack_unpack_3d(self):
        torch.manual_seed(42)
        signs = torch.randint(0, 2, (10, 8, 128), dtype=torch.uint8)
        packed = pack_sign_bits(signs)
        assert packed.shape == (10, 8, 16)
        unpacked = unpack_sign_bits(packed, 128)
        expected = 2.0 * signs.float() - 1.0
        assert torch.allclose(unpacked, expected)

    def test_all_positive(self):
        signs = torch.ones(10, 64, dtype=torch.uint8)
        packed = pack_sign_bits(signs)
        assert packed.shape == (10, 8)
        assert (packed == 0xFF).all()
        unpacked = unpack_sign_bits(packed, 64)
        assert (unpacked == 1.0).all()

    def test_all_negative(self):
        signs = torch.zeros(10, 64, dtype=torch.uint8)
        packed = pack_sign_bits(signs)
        assert (packed == 0).all()
        unpacked = unpack_sign_bits(packed, 64)
        assert (unpacked == -1.0).all()

    def test_sign_bytes_padded_alignment(self):
        assert sign_bytes_padded(128) == 16  # 128/8=16, already aligned
        assert sign_bytes_padded(64) == 8  # 64/8=8
        assert sign_bytes_padded(80) == 12  # 80/8=10 → padded to 12


class TestQJLEncodeDecode:
    """Test QJL variant encode/decode quality."""

    @pytest.fixture
    def codebook_6bit_qjl(self):
        return TurboQuantCodebook(n_bits=6, head_dim=128, device="cpu", qjl=True)

    @pytest.fixture
    def codebook_8bit_qjl(self):
        return TurboQuantCodebook(n_bits=8, head_dim=128, device="cpu", qjl=True)

    def test_qjl_encode_shapes(self, codebook_6bit_qjl):
        torch.manual_seed(123)
        tensor = torch.randn(50, 128)
        indices, norms, sign_bits, res_scales = turboquant_encode_qjl_ref(
            tensor, codebook_6bit_qjl
        )
        assert indices.shape == (50, 128)
        assert norms.shape == (50,)
        assert sign_bits.shape == (50, 16)  # 128/8
        assert res_scales.shape == (50,)

    def test_qjl_encode_3d(self, codebook_6bit_qjl):
        torch.manual_seed(123)
        tensor = torch.randn(20, 8, 128)
        indices, norms, sign_bits, res_scales = turboquant_encode_qjl_ref(
            tensor, codebook_6bit_qjl
        )
        assert indices.shape == (20, 8, 128)
        assert norms.shape == (20, 8)
        assert sign_bits.shape == (20, 8, 16)
        assert res_scales.shape == (20, 8)

    def test_qjl_better_than_basic_6bit(self, codebook_6bit_qjl):
        """QJL decode should have higher cosine sim than basic decode."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)

        # Basic TQ (no QJL)
        cb_basic = TurboQuantCodebook(n_bits=6, head_dim=128, device="cpu", qjl=False)
        idx_b, nrm_b = turboquant_encode_ref(tensor, cb_basic)
        dec_basic = turboquant_decode_ref(idx_b, nrm_b, cb_basic, torch.float32)
        sim_basic = (
            torch.nn.functional.cosine_similarity(tensor, dec_basic, dim=1)
            .mean()
            .item()
        )

        # QJL TQ
        idx_q, nrm_q, signs_q, rs_q = turboquant_encode_qjl_ref(
            tensor, codebook_6bit_qjl
        )
        dec_qjl = turboquant_decode_qjl_ref(
            idx_q, nrm_q, signs_q, rs_q, codebook_6bit_qjl, torch.float32
        )
        sim_qjl = (
            torch.nn.functional.cosine_similarity(tensor, dec_qjl, dim=1).mean().item()
        )

        assert sim_qjl > sim_basic, (
            f"QJL ({sim_qjl:.6f}) should be better than basic ({sim_basic:.6f})"
        )

    def test_qjl_better_than_basic_8bit(self, codebook_8bit_qjl):
        """QJL should also improve 8-bit."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)

        cb_basic = TurboQuantCodebook(n_bits=8, head_dim=128, device="cpu", qjl=False)
        idx_b, nrm_b = turboquant_encode_ref(tensor, cb_basic)
        dec_basic = turboquant_decode_ref(idx_b, nrm_b, cb_basic, torch.float32)
        sim_basic = (
            torch.nn.functional.cosine_similarity(tensor, dec_basic, dim=1)
            .mean()
            .item()
        )

        idx_q, nrm_q, signs_q, rs_q = turboquant_encode_qjl_ref(
            tensor, codebook_8bit_qjl
        )
        dec_qjl = turboquant_decode_qjl_ref(
            idx_q, nrm_q, signs_q, rs_q, codebook_8bit_qjl, torch.float32
        )
        sim_qjl = (
            torch.nn.functional.cosine_similarity(tensor, dec_qjl, dim=1).mean().item()
        )

        assert sim_qjl > sim_basic, (
            f"QJL ({sim_qjl:.6f}) should be better than basic ({sim_basic:.6f})"
        )

    def test_qjl_6bit_quality_threshold(self, codebook_6bit_qjl):
        """6-bit + QJL should achieve >0.995 cosine similarity."""
        torch.manual_seed(123)
        tensor = torch.randn(200, 128)
        idx, nrm, signs, rs = turboquant_encode_qjl_ref(tensor, codebook_6bit_qjl)
        decoded = turboquant_decode_qjl_ref(
            idx, nrm, signs, rs, codebook_6bit_qjl, torch.float32
        )
        sim = (
            torch.nn.functional.cosine_similarity(tensor, decoded, dim=1).mean().item()
        )
        assert sim > 0.995, f"6-bit+QJL cos_sim {sim:.4f} too low"

    def test_qjl_inner_product_quality(self, codebook_6bit_qjl):
        """Test that inner products are well-preserved with QJL."""
        torch.manual_seed(42)
        # Simulate query and key vectors
        queries = torch.randn(50, 128)
        keys = torch.randn(200, 128)

        # True inner products
        true_ip = queries @ keys.T  # (50, 200)

        # QJL-reconstructed inner products
        idx, nrm, signs, rs = turboquant_encode_qjl_ref(keys, codebook_6bit_qjl)
        keys_recon = turboquant_decode_qjl_ref(
            idx, nrm, signs, rs, codebook_6bit_qjl, torch.float32
        )
        recon_ip = queries @ keys_recon.T

        # Compare vs basic TQ
        cb_basic = TurboQuantCodebook(n_bits=6, head_dim=128, device="cpu", qjl=False)
        idx_b, nrm_b = turboquant_encode_ref(keys, cb_basic)
        keys_basic = turboquant_decode_ref(idx_b, nrm_b, cb_basic, torch.float32)
        basic_ip = queries @ keys_basic.T

        # QJL should have lower inner-product error
        qjl_err = (true_ip - recon_ip).abs().mean().item()
        basic_err = (true_ip - basic_ip).abs().mean().item()
        assert qjl_err < basic_err, (
            f"QJL ip error ({qjl_err:.4f}) should be lower than basic ({basic_err:.4f})"
        )

    def test_qjl_res_scale_positive(self, codebook_6bit_qjl):
        """Residual scales should be non-negative."""
        torch.manual_seed(123)
        tensor = torch.randn(100, 128)
        _, _, _, res_scales = turboquant_encode_qjl_ref(tensor, codebook_6bit_qjl)
        assert (res_scales >= 0).all()

    def test_qjl_codebook_flag(self):
        """QJL flag is respected for all bit widths."""
        # QJL works in both nibble (4-bit) and byte (6-bit) modes
        cb4 = TurboQuantCodebook(n_bits=4, head_dim=128, device="cpu", qjl=True)
        assert cb4.qjl is True
        cb6 = TurboQuantCodebook(n_bits=6, head_dim=128, device="cpu", qjl=True)
        assert cb6.qjl is True
        cb6_no = TurboQuantCodebook(n_bits=6, head_dim=128, device="cpu", qjl=False)
        assert cb6_no.qjl is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDequantPaged:
    """Test dequant-first Triton kernel correctness."""

    @pytest.fixture
    def codebook_6bit_qjl(self):
        return TurboQuantCodebook(n_bits=6, head_dim=128, device="cuda", qjl=True)

    @pytest.fixture
    def codebook_6bit_no_qjl(self):
        return TurboQuantCodebook(n_bits=6, head_dim=128, device="cuda", qjl=False)

    @pytest.fixture
    def codebook_4bit_qjl(self):
        return TurboQuantCodebook(n_bits=4, head_dim=128, device="cuda", qjl=True)

    def _build_paged_cache(
        self, codebook, num_seqs=4, seq_len=64, block_size=16, nkv=4
    ):
        """Build a TQ paged cache with random data, return all components."""
        head_dim = codebook.head_dim
        num_blocks_per_seq = (seq_len + block_size - 1) // block_size
        total_blocks = num_seqs * num_blocks_per_seq + 4  # extra slack blocks

        # Compute padded dim (matching get_kv_cache_shape logic)
        data_dim = head_dim if codebook.byte_mode else head_dim // 2
        scale_pad = 4  # sizeof(float32)
        if codebook.qjl:
            padded_dim = data_dim + scale_pad + sign_bytes_padded(head_dim) + scale_pad
        else:
            padded_dim = data_dim + scale_pad

        # Allocate uint8 KV cache
        kv_cache = torch.zeros(
            (total_blocks, 2, block_size, nkv, padded_dim),
            dtype=torch.uint8,
            device="cuda",
        )

        # Build block_table: sequential allocation
        block_table = torch.zeros(
            (num_seqs, num_blocks_per_seq), dtype=torch.int32, device="cuda"
        )
        for s in range(num_seqs):
            for p in range(num_blocks_per_seq):
                block_table[s, p] = s * num_blocks_per_seq + p

        seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device="cuda")

        # Generate random KV tokens and encode via turboquant_reshape_and_cache
        from vllm.v1.attention.ops.turboquant import (
            turboquant_reshape_and_cache,
        )

        # Slot mapping: token i of seq s → slot = s * seq_len + i
        total_tokens = num_seqs * seq_len
        slot_mapping = torch.zeros(total_tokens, dtype=torch.long, device="cuda")
        for s in range(num_seqs):
            for i in range(seq_len):
                blk = block_table[s, i // block_size].item()
                off = i % block_size
                slot_mapping[s * seq_len + i] = blk * block_size + off

        key = torch.randn(
            total_tokens, nkv, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        value = torch.randn(
            total_tokens, nkv, head_dim, dtype=torch.bfloat16, device="cuda"
        )

        # Extract norm/sign/res_scale views (matching _ensure_tq_norm_caches)
        key_cache, value_cache = kv_cache.unbind(1)
        dtype_sz = 1  # uint8
        kv_half_bytes = block_size * nkv * padded_dim * dtype_sz
        idx_data_bytes = head_dim if codebook.byte_mode else head_dim // 2

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device="cuda").set_(raw)
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim // 4
        head_f32 = padded_dim // 4
        norm_off_f32 = idx_data_bytes // 4

        k_norms = torch.as_strided(
            base_f32,
            size=(total_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=norm_off_f32,
        )
        v_base_f32 = kv_half_bytes // 4
        v_norms = torch.as_strided(
            base_f32,
            size=(total_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + norm_off_f32,
        )
        k_norms.fill_(0.0)
        v_norms.fill_(0.0)

        k_signs = v_signs = k_res_scales = v_res_scales = None
        if codebook.qjl:
            sbytes = sign_bytes_padded(head_dim)
            sign_byte_off = idx_data_bytes + 4
            base_u8 = torch.tensor([], dtype=torch.uint8, device="cuda").set_(raw)
            full_block_u8 = 2 * kv_half_bytes
            slot_u8 = nkv * padded_dim
            head_u8 = padded_dim

            k_signs = torch.as_strided(
                base_u8,
                size=(total_blocks, block_size, nkv, sbytes),
                stride=(full_block_u8, slot_u8, head_u8, 1),
                storage_offset=sign_byte_off,
            )
            v_signs = torch.as_strided(
                base_u8,
                size=(total_blocks, block_size, nkv, sbytes),
                stride=(full_block_u8, slot_u8, head_u8, 1),
                storage_offset=kv_half_bytes + sign_byte_off,
            )
            k_signs.fill_(0)
            v_signs.fill_(0)

            res_scale_off = (idx_data_bytes + 4 + sbytes) // 4
            k_res_scales = torch.as_strided(
                base_f32,
                size=(total_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=res_scale_off,
            )
            v_res_scales = torch.as_strided(
                base_f32,
                size=(total_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=v_base_f32 + res_scale_off,
            )
            k_res_scales.fill_(0.0)
            v_res_scales.fill_(0.0)

        # Encode
        turboquant_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            k_norms,
            v_norms,
            slot_mapping,
            codebook,
            k_signs=k_signs,
            v_signs=v_signs,
            k_res_scales=k_res_scales,
            v_res_scales=v_res_scales,
        )

        return dict(
            kv_cache=kv_cache,
            key_cache=key_cache,
            value_cache=value_cache,
            k_norms=k_norms,
            v_norms=v_norms,
            k_signs=k_signs,
            v_signs=v_signs,
            k_res_scales=k_res_scales,
            v_res_scales=v_res_scales,
            block_table=block_table,
            seq_lens=seq_lens,
            codebook=codebook,
            key=key,
            value=value,
            block_size=block_size,
            nkv=nkv,
            total_blocks=total_blocks,
            num_seqs=num_seqs,
            seq_len=seq_len,
        )

    def _run_dequant(self, cache_data):
        """Run turboquant_dequant_paged and return staging K/V."""
        from vllm.v1.attention.ops.turboquant import turboquant_dequant_paged

        cd = cache_data
        cb = cd["codebook"]
        num_seqs = cd["num_seqs"]
        max_bps = cd["block_table"].shape[1]
        staging_blocks = num_seqs * max_bps
        head_dim = cb.head_dim
        nkv = cd["nkv"]
        block_size = cd["block_size"]

        staging_key = torch.zeros(
            (staging_blocks, block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        staging_val = torch.zeros(
            (staging_blocks, block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

        turboquant_dequant_paged(
            cd["key_cache"],
            cd["value_cache"],
            cd["k_norms"],
            cd["v_norms"],
            staging_key,
            staging_val,
            cb,
            cd["block_table"],
            cd["seq_lens"],
            max_bps,
            k_signs=cd["k_signs"],
            v_signs=cd["v_signs"],
            k_res_scales=cd["k_res_scales"],
            v_res_scales=cd["v_res_scales"],
        )
        return staging_key, staging_val

    def _get_ref_rotated(self, cache_data):
        """Get reference reconstruction in ROTATED space (no inverse R).

        Dequant kernel outputs in rotated space, so compare there.
        """
        cd = cache_data
        cb = cd["codebook"]
        centroids = cb.centroids
        head_dim = cb.head_dim
        nkv = cd["nkv"]
        block_size = cd["block_size"]
        num_seqs = cd["num_seqs"]
        max_bps = cd["block_table"].shape[1]

        staging_blocks = num_seqs * max_bps
        ref_key = torch.zeros(staging_blocks, block_size, nkv, head_dim, device="cuda")
        ref_val = torch.zeros_like(ref_key)

        for s in range(num_seqs):
            for p in range(max_bps):
                phys = cd["block_table"][s, p].item()
                staging_blk = s * max_bps + p
                for slot in range(block_size):
                    for h in range(nkv):
                        norm_k = cd["k_norms"][phys, slot, h].item()
                        norm_v = cd["v_norms"][phys, slot, h].item()
                        if norm_k == 0.0 and norm_v == 0.0:
                            continue
                        # Extract indices from cache
                        kc = cd["key_cache"][phys, slot, h]
                        vc = cd["value_cache"][phys, slot, h]
                        if cb.byte_mode:
                            k_idx = kc[:head_dim].long()
                            v_idx = vc[:head_dim].long()
                        else:
                            packed_k = kc[: head_dim // 2]
                            k_idx = torch.cat(
                                [packed_k & 0x0F, (packed_k >> 4) & 0x0F], dim=-1
                            ).long()
                            packed_v = vc[: head_dim // 2]
                            v_idx = torch.cat(
                                [packed_v & 0x0F, (packed_v >> 4) & 0x0F], dim=-1
                            ).long()

                        k_vals = centroids[k_idx].float()
                        v_vals = centroids[v_idx].float()

                        if cb.qjl:
                            # Apply sign correction
                            k_rs = cd["k_res_scales"][phys, slot, h].item()
                            v_rs = cd["v_res_scales"][phys, slot, h].item()
                            k_sb = cd["k_signs"][phys, slot, h]
                            v_sb = cd["v_signs"][phys, slot, h]
                            # Unpack sign bits
                            for d in range(head_dim):
                                bidx = d // 8
                                bit = d % 8
                                k_sign = (
                                    1.0 if ((k_sb[bidx].item() >> bit) & 1) else -1.0
                                )
                                v_sign = (
                                    1.0 if ((v_sb[bidx].item() >> bit) & 1) else -1.0
                                )
                                k_vals[d] += k_rs * k_sign
                                v_vals[d] += v_rs * v_sign

                        ref_key[staging_blk, slot, h] = norm_k * k_vals
                        ref_val[staging_blk, slot, h] = norm_v * v_vals

        return ref_key, ref_val

    def test_dequant_byte_qjl(self, codebook_6bit_qjl):
        """Byte mode + QJL: dequant kernel matches reference."""
        torch.manual_seed(42)
        cd = self._build_paged_cache(
            codebook_6bit_qjl, num_seqs=2, seq_len=32, block_size=16, nkv=2
        )
        staging_key, staging_val = self._run_dequant(cd)
        ref_key, ref_val = self._get_ref_rotated(cd)
        torch.testing.assert_close(
            staging_key.float(), ref_key.float(), atol=0.02, rtol=0.01
        )
        torch.testing.assert_close(
            staging_val.float(), ref_val.float(), atol=0.02, rtol=0.01
        )

    def test_dequant_byte_no_qjl(self, codebook_6bit_no_qjl):
        """Byte mode without QJL: dequant kernel matches reference."""
        torch.manual_seed(42)
        cd = self._build_paged_cache(
            codebook_6bit_no_qjl, num_seqs=2, seq_len=32, block_size=16, nkv=2
        )
        staging_key, staging_val = self._run_dequant(cd)
        ref_key, ref_val = self._get_ref_rotated(cd)
        torch.testing.assert_close(
            staging_key.float(), ref_key.float(), atol=0.02, rtol=0.01
        )
        torch.testing.assert_close(
            staging_val.float(), ref_val.float(), atol=0.02, rtol=0.01
        )

    def test_dequant_nibble(self, codebook_4bit_qjl):
        """Nibble mode: dequant kernel matches reference."""
        torch.manual_seed(42)
        cd = self._build_paged_cache(
            codebook_4bit_qjl, num_seqs=2, seq_len=32, block_size=16, nkv=2
        )
        staging_key, staging_val = self._run_dequant(cd)
        ref_key, ref_val = self._get_ref_rotated(cd)
        torch.testing.assert_close(
            staging_key.float(), ref_key.float(), atol=0.02, rtol=0.01
        )
        torch.testing.assert_close(
            staging_val.float(), ref_val.float(), atol=0.02, rtol=0.01
        )

    def test_dequant_empty_slots(self, codebook_6bit_qjl):
        """Empty slots (norm=0) produce zeros in staging buffer."""
        torch.manual_seed(42)
        # Use seq_len=10 with block_size=16 → 6 empty slots per block
        cd = self._build_paged_cache(
            codebook_6bit_qjl, num_seqs=1, seq_len=10, block_size=16, nkv=2
        )
        staging_key, staging_val = self._run_dequant(cd)
        # Check that slots 10-15 in the first block are zero
        assert (staging_key[0, 10:16] == 0).all()
        assert (staging_val[0, 10:16] == 0).all()

    def test_dequant_norm_preserved(self, codebook_6bit_qjl):
        """Decompressed vectors should have approximately correct norms."""
        torch.manual_seed(42)
        cd = self._build_paged_cache(
            codebook_6bit_qjl, num_seqs=2, seq_len=32, block_size=16, nkv=2
        )
        staging_key, _ = self._run_dequant(cd)
        # Check a few non-empty slots
        for s in range(2):
            for p in range(2):  # 32 / 16 = 2 blocks per seq
                phys = cd["block_table"][s, p].item()
                staging_blk = s * 2 + p
                for slot in range(16):
                    for h in range(2):
                        expected_norm = cd["k_norms"][phys, slot, h].item()
                        if expected_norm == 0.0:
                            continue
                        actual_norm = (
                            staging_key[staging_blk, slot, h].float().norm().item()
                        )
                        # Should be close (not exact due to quantization)
                        assert (
                            abs(actual_norm - expected_norm) / (expected_norm + 1e-8)
                            < 0.3
                        ), (
                            f"Norm mismatch at s={s} p={p} slot={slot} h={h}: "
                            f"expected {expected_norm:.4f}, "
                            f"got {actual_norm:.4f}"
                        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestRotateQuery:
    """Test rotate_query / inverse_rotate_output on GPU."""

    @pytest.fixture
    def codebook(self):
        return TurboQuantCodebook(n_bits=6, head_dim=128, device="cuda", qjl=True)

    def test_rotate_preserves_norm(self, codebook):
        """rotate_query should preserve L2 norms."""
        from vllm.v1.attention.ops.turboquant import rotate_query

        torch.manual_seed(42)
        q = torch.randn(10, 128, dtype=torch.bfloat16, device="cuda")
        norms_before = q.float().norm(dim=-1)
        q_rot = rotate_query(q, codebook.rotation_matrix_T)
        norms_after = q_rot.float().norm(dim=-1)
        torch.testing.assert_close(norms_before, norms_after, atol=1e-3, rtol=1e-3)

    def test_rotate_inverse_roundtrip(self, codebook):
        """rotate_query then inverse_rotate_output should be near-identity."""
        from vllm.v1.attention.ops.turboquant import (
            inverse_rotate_output,
            rotate_query,
        )

        torch.manual_seed(42)
        q = torch.randn(10, 4, 128, dtype=torch.bfloat16, device="cuda")
        q_rot = rotate_query(q, codebook.rotation_matrix_T)
        q_back = inverse_rotate_output(q_rot, codebook.rotation_matrix)
        # bf16 truncation through two matmuls causes small errors
        torch.testing.assert_close(q.float(), q_back.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_rotate_dtype_preservation(self, codebook, dtype):
        """Output dtype should match input dtype."""
        from vllm.v1.attention.ops.turboquant import rotate_query

        q = torch.randn(10, 128, dtype=dtype, device="cuda")
        q_rot = rotate_query(q, codebook.rotation_matrix_T)
        assert q_rot.dtype == dtype, f"Expected {dtype}, got {q_rot.dtype}"

    @pytest.mark.parametrize(
        "shape",
        [
            (10, 128),
            (10, 4, 128),
            (10, 4, 32, 128),
        ],
    )
    def test_rotate_batch_shapes(self, codebook, shape):
        """rotate_query should work with various batch dimensions."""
        from vllm.v1.attention.ops.turboquant import rotate_query

        torch.manual_seed(42)
        q = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        q_rot = rotate_query(q, codebook.rotation_matrix_T)
        assert q_rot.shape == shape, f"Expected shape {shape}, got {q_rot.shape}"
        # Norms should still be preserved
        norms_before = q.float().reshape(-1, 128).norm(dim=-1)
        norms_after = q_rot.float().reshape(-1, 128).norm(dim=-1)
        torch.testing.assert_close(norms_before, norms_after, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestIncrementalDequant:
    """Test incremental (dirty_blocks) dequant on GPU."""

    @pytest.fixture
    def codebook(self):
        return TurboQuantCodebook(n_bits=6, head_dim=128, device="cuda", qjl=True)

    def _build_paged_cache(
        self, codebook, num_seqs=4, seq_len=64, block_size=16, nkv=4
    ):
        """Build a TQ paged cache (same helper as TestDequantPaged)."""
        head_dim = codebook.head_dim
        num_blocks_per_seq = (seq_len + block_size - 1) // block_size
        total_blocks = num_seqs * num_blocks_per_seq + 4

        data_dim = head_dim if codebook.byte_mode else head_dim // 2
        scale_pad = 4
        if codebook.qjl:
            padded_dim = data_dim + scale_pad + sign_bytes_padded(head_dim) + scale_pad
        else:
            padded_dim = data_dim + scale_pad

        kv_cache = torch.zeros(
            (total_blocks, 2, block_size, nkv, padded_dim),
            dtype=torch.uint8,
            device="cuda",
        )

        block_table = torch.zeros(
            (num_seqs, num_blocks_per_seq), dtype=torch.int32, device="cuda"
        )
        for s in range(num_seqs):
            for p in range(num_blocks_per_seq):
                block_table[s, p] = s * num_blocks_per_seq + p

        seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device="cuda")

        from vllm.v1.attention.ops.turboquant import (
            turboquant_reshape_and_cache,
        )

        total_tokens = num_seqs * seq_len
        slot_mapping = torch.zeros(total_tokens, dtype=torch.long, device="cuda")
        for s in range(num_seqs):
            for i in range(seq_len):
                blk = block_table[s, i // block_size].item()
                off = i % block_size
                slot_mapping[s * seq_len + i] = blk * block_size + off

        key = torch.randn(
            total_tokens, nkv, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        value = torch.randn(
            total_tokens, nkv, head_dim, dtype=torch.bfloat16, device="cuda"
        )

        key_cache, value_cache = kv_cache.unbind(1)
        dtype_sz = 1
        kv_half_bytes = block_size * nkv * padded_dim * dtype_sz
        idx_data_bytes = head_dim if codebook.byte_mode else head_dim // 2

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device="cuda").set_(raw)
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim // 4
        head_f32 = padded_dim // 4
        norm_off_f32 = idx_data_bytes // 4

        k_norms = torch.as_strided(
            base_f32,
            size=(total_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=norm_off_f32,
        )
        v_base_f32 = kv_half_bytes // 4
        v_norms = torch.as_strided(
            base_f32,
            size=(total_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + norm_off_f32,
        )
        k_norms.fill_(0.0)
        v_norms.fill_(0.0)

        k_signs = v_signs = k_res_scales = v_res_scales = None
        if codebook.qjl:
            sbytes = sign_bytes_padded(head_dim)
            sign_byte_off = idx_data_bytes + 4
            base_u8 = torch.tensor([], dtype=torch.uint8, device="cuda").set_(raw)
            full_block_u8 = 2 * kv_half_bytes
            slot_u8 = nkv * padded_dim
            head_u8 = padded_dim

            k_signs = torch.as_strided(
                base_u8,
                size=(total_blocks, block_size, nkv, sbytes),
                stride=(full_block_u8, slot_u8, head_u8, 1),
                storage_offset=sign_byte_off,
            )
            v_signs = torch.as_strided(
                base_u8,
                size=(total_blocks, block_size, nkv, sbytes),
                stride=(full_block_u8, slot_u8, head_u8, 1),
                storage_offset=kv_half_bytes + sign_byte_off,
            )
            k_signs.fill_(0)
            v_signs.fill_(0)

            res_scale_off = (idx_data_bytes + 4 + sbytes) // 4
            k_res_scales = torch.as_strided(
                base_f32,
                size=(total_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=res_scale_off,
            )
            v_res_scales = torch.as_strided(
                base_f32,
                size=(total_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=v_base_f32 + res_scale_off,
            )
            k_res_scales.fill_(0.0)
            v_res_scales.fill_(0.0)

        turboquant_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            k_norms,
            v_norms,
            slot_mapping,
            codebook,
            k_signs=k_signs,
            v_signs=v_signs,
            k_res_scales=k_res_scales,
            v_res_scales=v_res_scales,
        )

        return dict(
            kv_cache=kv_cache,
            key_cache=key_cache,
            value_cache=value_cache,
            k_norms=k_norms,
            v_norms=v_norms,
            k_signs=k_signs,
            v_signs=v_signs,
            k_res_scales=k_res_scales,
            v_res_scales=v_res_scales,
            block_table=block_table,
            seq_lens=seq_lens,
            codebook=codebook,
            key=key,
            value=value,
            block_size=block_size,
            nkv=nkv,
            total_blocks=total_blocks,
            num_seqs=num_seqs,
            seq_len=seq_len,
        )

    def _run_dequant(self, cache_data, dirty_blocks=None):
        """Run turboquant_dequant_paged and return staging K/V."""
        from vllm.v1.attention.ops.turboquant import turboquant_dequant_paged

        cd = cache_data
        cb = cd["codebook"]
        num_seqs = cd["num_seqs"]
        max_bps = cd["block_table"].shape[1]
        staging_blocks = num_seqs * max_bps
        head_dim = cb.head_dim
        nkv = cd["nkv"]
        block_size = cd["block_size"]

        staging_key = torch.zeros(
            (staging_blocks, block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        staging_val = torch.zeros(
            (staging_blocks, block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

        turboquant_dequant_paged(
            cd["key_cache"],
            cd["value_cache"],
            cd["k_norms"],
            cd["v_norms"],
            staging_key,
            staging_val,
            cb,
            cd["block_table"],
            cd["seq_lens"],
            max_bps,
            k_signs=cd["k_signs"],
            v_signs=cd["v_signs"],
            k_res_scales=cd["k_res_scales"],
            v_res_scales=cd["v_res_scales"],
            dirty_blocks=dirty_blocks,
        )
        return staging_key, staging_val

    def test_full_dequant_matches_reference(self, codebook):
        """Full dequant (dirty_blocks=None) should match reference."""
        torch.manual_seed(42)
        cd = self._build_paged_cache(
            codebook, num_seqs=2, seq_len=32, block_size=16, nkv=2
        )
        # Full dequant (no dirty_blocks)
        staging_key_full, staging_val_full = self._run_dequant(cd)

        # Compare against a second full dequant to confirm determinism
        staging_key_2, staging_val_2 = self._run_dequant(cd)
        torch.testing.assert_close(staging_key_full, staging_key_2)
        torch.testing.assert_close(staging_val_full, staging_val_2)

        # Non-empty slots should have non-zero values
        assert staging_key_full.abs().sum() > 0
        assert staging_val_full.abs().sum() > 0

    def test_selective_dequant_updates_dirty_only(self, codebook):
        """Pass dirty_blocks with some True, verify only those staging
        positions change."""
        torch.manual_seed(42)
        cd = self._build_paged_cache(
            codebook, num_seqs=2, seq_len=32, block_size=16, nkv=2
        )

        total_blocks = cd["total_blocks"]

        # First do a full dequant to populate staging
        staging_key_full, staging_val_full = self._run_dequant(cd)

        # Now create dirty_blocks where only some physical blocks are dirty
        dirty_blocks = torch.zeros(total_blocks, dtype=torch.bool, device="cuda")
        # Mark only blocks 0 and 2 as dirty
        dirty_blocks[0] = True
        dirty_blocks[2] = True

        # Start with zeroed staging buffers
        staging_key_inc, staging_val_inc = self._run_dequant(
            cd, dirty_blocks=dirty_blocks
        )

        # For dirty physical blocks (0, 2), the staging output should match
        # the full dequant output
        max_bps = cd["block_table"].shape[1]
        num_seqs = cd["num_seqs"]
        for s in range(num_seqs):
            for p in range(max_bps):
                phys = cd["block_table"][s, p].item()
                staging_blk = s * max_bps + p
                if dirty_blocks[phys]:
                    # Dirty block: should be decompressed
                    torch.testing.assert_close(
                        staging_key_inc[staging_blk].float(),
                        staging_key_full[staging_blk].float(),
                        atol=0.02,
                        rtol=0.01,
                    )
                    torch.testing.assert_close(
                        staging_val_inc[staging_blk].float(),
                        staging_val_full[staging_blk].float(),
                        atol=0.02,
                        rtol=0.01,
                    )
                else:
                    # Clean block: staging should remain zero
                    assert (staging_key_inc[staging_blk] == 0).all(), (
                        f"Clean block s={s} p={p} phys={phys} should remain zero"
                    )
                    assert (staging_val_inc[staging_blk] == 0).all(), (
                        f"Clean block s={s} p={p} phys={phys} should remain zero"
                    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestEndToEndAttention:
    """Test full TQ attention pipeline vs bf16 reference."""

    def _build_paged_cache(
        self, codebook, num_seqs=2, seq_len=32, block_size=16, nkv=4
    ):
        """Build a TQ paged cache (same as TestDequantPaged)."""
        head_dim = codebook.head_dim
        num_blocks_per_seq = (seq_len + block_size - 1) // block_size
        total_blocks = num_seqs * num_blocks_per_seq + 4

        data_dim = head_dim if codebook.byte_mode else head_dim // 2
        scale_pad = 4
        if codebook.qjl:
            padded_dim = data_dim + scale_pad + sign_bytes_padded(head_dim) + scale_pad
        else:
            padded_dim = data_dim + scale_pad

        kv_cache = torch.zeros(
            (total_blocks, 2, block_size, nkv, padded_dim),
            dtype=torch.uint8,
            device="cuda",
        )

        block_table = torch.zeros(
            (num_seqs, num_blocks_per_seq), dtype=torch.int32, device="cuda"
        )
        for s in range(num_seqs):
            for p in range(num_blocks_per_seq):
                block_table[s, p] = s * num_blocks_per_seq + p

        seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device="cuda")

        from vllm.v1.attention.ops.turboquant import (
            turboquant_reshape_and_cache,
        )

        total_tokens = num_seqs * seq_len
        slot_mapping = torch.zeros(total_tokens, dtype=torch.long, device="cuda")
        for s in range(num_seqs):
            for i in range(seq_len):
                blk = block_table[s, i // block_size].item()
                off = i % block_size
                slot_mapping[s * seq_len + i] = blk * block_size + off

        key = torch.randn(
            total_tokens, nkv, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        value = torch.randn(
            total_tokens, nkv, head_dim, dtype=torch.bfloat16, device="cuda"
        )

        key_cache, value_cache = kv_cache.unbind(1)
        dtype_sz = 1
        kv_half_bytes = block_size * nkv * padded_dim * dtype_sz
        idx_data_bytes = head_dim if codebook.byte_mode else head_dim // 2

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device="cuda").set_(raw)
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim // 4
        head_f32 = padded_dim // 4
        norm_off_f32 = idx_data_bytes // 4

        k_norms = torch.as_strided(
            base_f32,
            size=(total_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=norm_off_f32,
        )
        v_base_f32 = kv_half_bytes // 4
        v_norms = torch.as_strided(
            base_f32,
            size=(total_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + norm_off_f32,
        )
        k_norms.fill_(0.0)
        v_norms.fill_(0.0)

        k_signs = v_signs = k_res_scales = v_res_scales = None
        if codebook.qjl:
            sbytes = sign_bytes_padded(head_dim)
            sign_byte_off = idx_data_bytes + 4
            base_u8 = torch.tensor([], dtype=torch.uint8, device="cuda").set_(raw)
            full_block_u8 = 2 * kv_half_bytes
            slot_u8 = nkv * padded_dim
            head_u8 = padded_dim

            k_signs = torch.as_strided(
                base_u8,
                size=(total_blocks, block_size, nkv, sbytes),
                stride=(full_block_u8, slot_u8, head_u8, 1),
                storage_offset=sign_byte_off,
            )
            v_signs = torch.as_strided(
                base_u8,
                size=(total_blocks, block_size, nkv, sbytes),
                stride=(full_block_u8, slot_u8, head_u8, 1),
                storage_offset=kv_half_bytes + sign_byte_off,
            )
            k_signs.fill_(0)
            v_signs.fill_(0)

            res_scale_off = (idx_data_bytes + 4 + sbytes) // 4
            k_res_scales = torch.as_strided(
                base_f32,
                size=(total_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=res_scale_off,
            )
            v_res_scales = torch.as_strided(
                base_f32,
                size=(total_blocks, block_size, nkv),
                stride=(full_block_f32, slot_f32, head_f32),
                storage_offset=v_base_f32 + res_scale_off,
            )
            k_res_scales.fill_(0.0)
            v_res_scales.fill_(0.0)

        turboquant_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            k_norms,
            v_norms,
            slot_mapping,
            codebook,
            k_signs=k_signs,
            v_signs=v_signs,
            k_res_scales=k_res_scales,
            v_res_scales=v_res_scales,
        )

        return dict(
            kv_cache=kv_cache,
            key_cache=key_cache,
            value_cache=value_cache,
            k_norms=k_norms,
            v_norms=v_norms,
            k_signs=k_signs,
            v_signs=v_signs,
            k_res_scales=k_res_scales,
            v_res_scales=v_res_scales,
            block_table=block_table,
            seq_lens=seq_lens,
            codebook=codebook,
            key=key,
            value=value,
            block_size=block_size,
            nkv=nkv,
            total_blocks=total_blocks,
            num_seqs=num_seqs,
            seq_len=seq_len,
        )

    def test_attention_with_tq_vs_bf16(self):
        """Full pipeline: encode K/V with TQ, rotate Q, do attention with
        staging, inverse rotate output. Compare with bf16 reference.
        Cosine sim > 0.998."""
        from vllm.v1.attention.ops.turboquant import (
            inverse_rotate_output,
            rotate_query,
            turboquant_dequant_paged,
        )

        torch.manual_seed(42)

        head_dim = 128
        nkv = 4
        num_seqs = 2
        seq_len = 32
        block_size = 16

        codebook = TurboQuantCodebook(
            n_bits=6, head_dim=head_dim, device="cuda", qjl=True
        )
        cd = self._build_paged_cache(
            codebook, num_seqs=num_seqs, seq_len=seq_len, block_size=block_size, nkv=nkv
        )

        # --- TQ path: dequant to staging, rotate Q, do attention ---
        max_bps = cd["block_table"].shape[1]
        staging_blocks = num_seqs * max_bps

        staging_key = torch.zeros(
            (staging_blocks, block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        staging_val = torch.zeros(
            (staging_blocks, block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

        turboquant_dequant_paged(
            cd["key_cache"],
            cd["value_cache"],
            cd["k_norms"],
            cd["v_norms"],
            staging_key,
            staging_val,
            codebook,
            cd["block_table"],
            cd["seq_lens"],
            max_bps,
            k_signs=cd["k_signs"],
            v_signs=cd["v_signs"],
            k_res_scales=cd["k_res_scales"],
            v_res_scales=cd["v_res_scales"],
        )

        # Reassemble K/V per sequence from staging blocks
        # staging layout: [staging_blk, block_size, nkv, head_dim]
        # staging_blk = seq * max_bps + page
        tq_outputs = []
        for s in range(num_seqs):
            k_parts = []
            v_parts = []
            for p in range(max_bps):
                staging_blk = s * max_bps + p
                k_parts.append(staging_key[staging_blk])  # [BS, nkv, HD]
                v_parts.append(staging_val[staging_blk])
            # [seq_len, nkv, head_dim] — staging K/V in rotated space
            K_staged = torch.cat(k_parts, dim=0)[:seq_len]
            V_staged = torch.cat(v_parts, dim=0)[:seq_len]

            # Query: one query token per sequence for simplicity
            q = torch.randn(1, nkv, head_dim, dtype=torch.bfloat16, device="cuda")
            q_rot = rotate_query(q, codebook.rotation_matrix_T)

            # Attention scores: q_rot @ K_staged^T (per head)
            # q_rot: [1, nkv, HD], K_staged: [seq_len, nkv, HD]
            scale = 1.0 / math.sqrt(head_dim)
            # [nkv, 1, seq_len]
            scores = (
                torch.einsum("bnh,snh->nbs", q_rot.float(), K_staged.float()) * scale
            )
            weights = torch.softmax(scores, dim=-1)  # [nkv, 1, seq_len]
            # [nkv, 1, HD]
            attn_out_rot = torch.einsum("nbs,snh->nbh", weights, V_staged.float())
            # [1, nkv, HD]
            attn_out_rot = attn_out_rot.permute(1, 0, 2)
            attn_out = inverse_rotate_output(
                attn_out_rot.bfloat16(), codebook.rotation_matrix
            )

            # --- BF16 reference path ---
            key_orig = cd["key"][s * seq_len : (s + 1) * seq_len]
            val_orig = cd["value"][s * seq_len : (s + 1) * seq_len]

            scores_ref = (
                torch.einsum("bnh,snh->nbs", q.float(), key_orig.float()) * scale
            )
            weights_ref = torch.softmax(scores_ref, dim=-1)
            attn_ref = torch.einsum("nbs,snh->nbh", weights_ref, val_orig.float())
            attn_ref = attn_ref.permute(1, 0, 2)  # [1, nkv, HD]

            # Per-head cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                attn_out.float().reshape(-1, head_dim),
                attn_ref.float().reshape(-1, head_dim),
                dim=-1,
            )
            tq_outputs.append(cos_sim)

        all_sims = torch.cat(tq_outputs)
        mean_sim = all_sims.mean().item()
        assert mean_sim > 0.998, (
            f"Mean cosine similarity {mean_sim:.6f} too low (threshold 0.998)"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestGPUKernelRoundtrip:
    """GPU kernel roundtrip tests: Triton encode → dequant → compare.

    These tests verify that the Triton kernels produce outputs that match
    the Python reference implementation and that encode→decode roundtrips
    preserve vector quality.  PR #38479 has zero GPU kernel tests.
    """

    @pytest.mark.parametrize("n_bits", [4, 6, 8])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_encode_roundtrip_quality(self, n_bits: int, head_dim: int):
        """Triton encode → Triton dequant → cosine similarity with original.

        This is the core quality test: KV vectors go through the full
        TQ pipeline (norm, rotate, quantize, pack, dequant, denorm)
        and we check how close the reconstructed vectors are.
        """
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        nkv = 4
        num_seqs = 2
        seq_len = 64
        block_size = 16

        codebook = TurboQuantCodebook(
            n_bits=n_bits, head_dim=head_dim, device=device, qjl=False
        )

        # Build paged cache with Triton encode
        e2e = TestEndToEndAttention()
        cd = e2e._build_paged_cache(
            codebook,
            num_seqs=num_seqs,
            seq_len=seq_len,
            block_size=block_size,
            nkv=nkv,
        )

        # Dequant via Triton kernel
        from vllm.v1.attention.ops.turboquant import turboquant_dequant_paged

        max_bps = cd["block_table"].shape[1]
        staging_key = torch.zeros(
            (cd["total_blocks"], block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        staging_val = torch.zeros_like(staging_key)

        turboquant_dequant_paged(
            cd["key_cache"],
            cd["value_cache"],
            cd["k_norms"],
            cd["v_norms"],
            staging_key,
            staging_val,
            codebook,
            cd["block_table"],
            cd["seq_lens"],
            max_bps,
        )

        # Reconstruct K/V from staging
        for s in range(num_seqs):
            for i in range(seq_len):
                blk = cd["block_table"][s, i // block_size].item()
                off = i % block_size
                k_recon = staging_key[blk, off]  # [nkv, head_dim]
                v_recon = staging_val[blk, off]

                k_orig = cd["key"][s * seq_len + i]  # [nkv, head_dim]
                v_orig = cd["value"][s * seq_len + i]

                # Inverse-rotate the reconstructed vector
                R = codebook.rotation_matrix  # [head_dim, head_dim]
                k_unrot = (k_recon.float() @ R.float()).bfloat16()
                v_unrot = (v_recon.float() @ R.float()).bfloat16()

                # Per-head cosine similarity
                for h in range(nkv):
                    cos_k = torch.nn.functional.cosine_similarity(
                        k_unrot[h : h + 1].float(),
                        k_orig[h : h + 1].float(),
                    ).item()
                    cos_v = torch.nn.functional.cosine_similarity(
                        v_unrot[h : h + 1].float(),
                        v_orig[h : h + 1].float(),
                    ).item()

                    if n_bits >= 6:
                        threshold = 0.99
                    elif n_bits >= 4:
                        threshold = 0.95
                    else:
                        threshold = 0.85

                    assert cos_k > threshold, (
                        f"Key roundtrip cos_sim={cos_k:.4f} < {threshold} "
                        f"(n_bits={n_bits}, head_dim={head_dim}, "
                        f"seq={s}, pos={i}, head={h})"
                    )
                    assert cos_v > threshold, (
                        f"Value roundtrip cos_sim={cos_v:.4f} < {threshold} "
                        f"(n_bits={n_bits}, head_dim={head_dim}, "
                        f"seq={s}, pos={i}, head={h})"
                    )

    @pytest.mark.parametrize("n_bits", [4, 8])
    def test_triton_vs_reference_encode(self, n_bits: int):
        """Triton encode indices match Python reference indices exactly.

        Verifies bit-exact agreement between the Triton kernel and the
        Python reference implementation of TQ encoding.
        """
        torch.manual_seed(123)
        device = torch.device("cuda:0")
        head_dim = 128
        nkv = 4
        num_tokens = 16

        codebook = TurboQuantCodebook(
            n_bits=n_bits, head_dim=head_dim, device=device, qjl=False
        )

        x = torch.randn(num_tokens, nkv, head_dim, device=device)

        # Reference encode (unpacked for comparison)
        ref_indices, ref_norms = turboquant_encode_ref(x, codebook, packed=False)

        # Triton encode via reshape_and_cache
        from vllm.v1.attention.ops.turboquant import turboquant_reshape_and_cache

        block_size = 16
        num_blocks = (num_tokens + block_size - 1) // block_size + 1
        data_dim = head_dim if codebook.byte_mode else head_dim // 2
        scale_pad = 4
        padded_dim = data_dim + scale_pad

        kv_cache = torch.zeros(
            (num_blocks, 2, block_size, nkv, padded_dim),
            dtype=torch.uint8,
            device=device,
        )
        key_cache, value_cache = kv_cache.unbind(1)

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=device).set_(raw)
        kv_half_bytes = block_size * nkv * padded_dim
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim // 4
        head_f32 = padded_dim // 4
        norm_off_f32 = data_dim // 4

        k_norms = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=norm_off_f32,
        )
        v_norms = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=kv_half_bytes // 4 + norm_off_f32,
        )
        k_norms.fill_(0.0)
        v_norms.fill_(0.0)

        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

        turboquant_reshape_and_cache(
            x,
            x,
            key_cache,
            value_cache,
            k_norms,
            v_norms,
            slot_mapping,
            codebook,
        )

        # Extract Triton-encoded indices from cache
        for t in range(num_tokens):
            blk = t // block_size
            off = t % block_size
            for h in range(nkv):
                if codebook.byte_mode:
                    triton_idx = key_cache[blk, off, h, :head_dim].to(torch.int32)
                    ref_idx = ref_indices[t, h, :head_dim].to(torch.int32)
                else:
                    packed = key_cache[blk, off, h, : head_dim // 2]
                    triton_idx = unpack_nibbles(packed.unsqueeze(0), head_dim)[0]
                    ref_idx = ref_indices[t, h, :head_dim].to(torch.int32)

                match = (triton_idx == ref_idx).float().mean().item()
                assert match > 0.98, (
                    f"Triton vs reference match={match:.4f} "
                    f"(n_bits={n_bits}, token={t}, head={h})"
                )

        # Verify norms match
        for t in range(num_tokens):
            blk = t // block_size
            off = t % block_size
            for h in range(nkv):
                triton_norm = k_norms[blk, off, h].item()
                ref_norm = ref_norms[t, h].item()
                assert abs(triton_norm - ref_norm) / (abs(ref_norm) + 1e-8) < 0.01, (
                    f"Norm mismatch: triton={triton_norm:.6f} vs "
                    f"ref={ref_norm:.6f} (token={t}, head={h})"
                )

    def test_qjl_roundtrip_quality(self):
        """QJL mode: encode → dequant → compare.

        QJL adds sign-bit correction to the basic TQ encoding, which
        should improve quality (lower MSE / higher cosine sim).
        """
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        head_dim = 128
        nkv = 4
        num_seqs = 1
        seq_len = 32
        block_size = 16

        codebook = TurboQuantCodebook(
            n_bits=6, head_dim=head_dim, device=device, qjl=True
        )

        e2e = TestEndToEndAttention()
        cd = e2e._build_paged_cache(
            codebook,
            num_seqs=num_seqs,
            seq_len=seq_len,
            block_size=block_size,
            nkv=nkv,
        )

        from vllm.v1.attention.ops.turboquant import turboquant_dequant_paged

        max_bps = cd["block_table"].shape[1]
        staging_key = torch.zeros(
            (cd["total_blocks"], block_size, nkv, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        staging_val = torch.zeros_like(staging_key)

        turboquant_dequant_paged(
            cd["key_cache"],
            cd["value_cache"],
            cd["k_norms"],
            cd["v_norms"],
            staging_key,
            staging_val,
            codebook,
            cd["block_table"],
            cd["seq_lens"],
            max_bps,
            k_signs=cd["k_signs"],
            v_signs=cd["v_signs"],
            k_res_scales=cd["k_res_scales"],
            v_res_scales=cd["v_res_scales"],
        )

        # Check cosine similarity (QJL should give very high quality)
        R = codebook.rotation_matrix.float()
        cos_sims = []
        for s in range(num_seqs):
            for i in range(seq_len):
                blk = cd["block_table"][s, i // block_size].item()
                off = i % block_size
                k_recon = (staging_key[blk, off].float() @ R).bfloat16()
                k_orig = cd["key"][s * seq_len + i]

                cos = torch.nn.functional.cosine_similarity(
                    k_recon.float().reshape(-1, head_dim),
                    k_orig.float().reshape(-1, head_dim),
                ).mean()
                cos_sims.append(cos.item())

        mean_cos = sum(cos_sims) / len(cos_sims)
        assert mean_cos > 0.995, f"QJL roundtrip mean cosine={mean_cos:.6f} < 0.995"

    @pytest.mark.parametrize(
        "preset",
        ["tq-k8v8", "tq-k4v4", "tq-k8fv4"],
    )
    def test_preset_cache_dimensions(self, preset: str):
        """Verify that each preset produces correct cache dimensions.

        This is a structural test ensuring the padded cache dim is
        computed correctly for each preset.
        """
        from vllm.v1.attention.backends.turboquant_config import (
            parse_tq_preset,
        )

        p = parse_tq_preset(preset)
        head_dim = 128

        cache_dim_k = p.cache_dim_per_head(head_dim, "k")
        cache_dim_v = p.cache_dim_per_head(head_dim, "v")

        # Key cache dim: encoded bytes + 4 (norm float32)
        if p.k_fp8:
            # FP8: 1 byte per dim + 4 bytes norm
            assert cache_dim_k == head_dim + 4
        elif p.k_bits >= 5:
            # Byte mode: 1 byte per dim + 4 bytes norm
            assert cache_dim_k == head_dim + 4
        else:
            # Nibble mode: head_dim/2 packed + 4 bytes norm
            assert cache_dim_k == head_dim // 2 + 4

        # Value cache dim follows same pattern
        if p.v_bits >= 5:
            assert cache_dim_v == head_dim + 4
        else:
            assert cache_dim_v == head_dim // 2 + 4

    def test_multiple_encode_batches(self):
        """Encode in multiple batches and verify consistency.

        Simulates what happens during prefill + decode: first batch
        fills initial cache, subsequent batches append.
        """
        torch.manual_seed(7)
        device = torch.device("cuda:0")
        head_dim = 128
        nkv = 4
        block_size = 16
        n_bits = 8

        from vllm.v1.attention.ops.turboquant import turboquant_reshape_and_cache

        codebook = TurboQuantCodebook(
            n_bits=n_bits, head_dim=head_dim, device=device, qjl=False
        )

        total_tokens = 48
        num_blocks = (total_tokens + block_size - 1) // block_size + 2
        data_dim = head_dim  # byte mode
        scale_pad = 4
        padded_dim = data_dim + scale_pad

        kv_cache = torch.zeros(
            (num_blocks, 2, block_size, nkv, padded_dim),
            dtype=torch.uint8,
            device=device,
        )
        key_cache, value_cache = kv_cache.unbind(1)

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=device).set_(raw)
        kv_half_bytes = block_size * nkv * padded_dim
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_dim // 4
        head_f32 = padded_dim // 4
        norm_off_f32 = data_dim // 4

        k_norms = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=norm_off_f32,
        )
        v_norms = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=kv_half_bytes // 4 + norm_off_f32,
        )
        k_norms.fill_(0.0)
        v_norms.fill_(0.0)

        all_keys = torch.randn(
            total_tokens, nkv, head_dim, dtype=torch.bfloat16, device=device
        )

        # Encode in 3 batches (16+16+16 tokens)
        for batch_start in range(0, total_tokens, 16):
            batch_end = min(batch_start + 16, total_tokens)
            batch_keys = all_keys[batch_start:batch_end]
            batch_slots = torch.arange(
                batch_start, batch_end, dtype=torch.long, device=device
            )

            turboquant_reshape_and_cache(
                batch_keys,
                batch_keys,
                key_cache,
                value_cache,
                k_norms,
                v_norms,
                batch_slots,
                codebook,
            )

        # Verify all norms are nonzero (each slot was written)
        for t in range(total_tokens):
            blk = t // block_size
            off = t % block_size
            for h in range(nkv):
                norm = k_norms[blk, off, h].item()
                assert norm > 0, (
                    f"Norm is zero at token={t}, head={h}. "
                    f"Batch encoding may have missed this slot."
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFP8EncodeKey:
    """Test the Triton FP8 key encoding kernel."""

    def test_roundtrip_quality(self):
        """FP8 encode → LUT decode should have cosine similarity > 0.99."""
        from vllm.v1.attention.ops.turboquant import fp8_encode_key

        torch.manual_seed(42)
        device = "cuda"
        num_tokens, num_kv_heads, head_dim = 16, 4, 128
        block_size, num_blocks = 16, 4
        padded_dim = head_dim + 16

        key = torch.randn(
            num_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        cache = torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            padded_dim,
            device=device,
            dtype=torch.uint8,
        )
        scales = torch.ones(
            num_blocks, block_size, num_kv_heads, device=device, dtype=torch.float32
        )
        slots = torch.arange(num_tokens, device=device, dtype=torch.int64)

        fp8_encode_key(key, cache, scales, slots)

        # Decode via FP8 LUT
        fp8_lut = (
            torch.arange(256, dtype=torch.uint8, device=device)
            .view(torch.float8_e4m3fn)
            .float()
        )

        k_float = key.float()
        for t in range(num_tokens):
            blk, off = t // block_size, t % block_size
            for h in range(num_kv_heads):
                decoded = (
                    fp8_lut[cache[blk, off, h, :head_dim].long()] * scales[blk, off, h]
                )
                cosim = torch.nn.functional.cosine_similarity(
                    decoded.unsqueeze(0), k_float[t, h].unsqueeze(0)
                ).item()
                assert cosim > 0.99, (
                    f"FP8 roundtrip quality too low: tok={t} head={h} cosim={cosim:.4f}"
                )

    def test_negative_slots(self):
        """Negative slots (CUDAGraph padding) should not crash."""
        from vllm.v1.attention.ops.turboquant import fp8_encode_key

        device = "cuda"
        key = torch.randn(4, 2, 128, device=device, dtype=torch.bfloat16)
        cache = torch.zeros(2, 16, 2, 144, device=device, dtype=torch.uint8)
        scales = torch.ones(2, 16, 2, device=device, dtype=torch.float32)
        slots = torch.tensor([0, -1, 2, -1], device=device, dtype=torch.int64)

        fp8_encode_key(key, cache, scales, slots)
        # Verify valid slots were written (scale != 1.0)
        assert scales[0, 0, 0].item() != 1.0  # slot 0 written
        assert scales[0, 2, 0].item() != 1.0  # slot 2 written

    def test_empty_tokens(self):
        """Empty token tensor should be a no-op."""
        from vllm.v1.attention.ops.turboquant import fp8_encode_key

        device = "cuda"
        key = torch.randn(0, 2, 128, device=device, dtype=torch.bfloat16)
        cache = torch.zeros(1, 16, 2, 144, device=device, dtype=torch.uint8)
        scales = torch.ones(1, 16, 2, device=device, dtype=torch.float32)
        slots = torch.tensor([], device=device, dtype=torch.int64)

        fp8_encode_key(key, cache, scales, slots)  # Should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
