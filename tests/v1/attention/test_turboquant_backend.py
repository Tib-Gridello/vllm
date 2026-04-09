# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the standalone TurboQuant attention backend."""

import pytest

from vllm.v1.attention.backends.turboquant_config import (
    is_tq_preset,
    parse_tq_preset,
)

# ============================================================================
# Preset Parsing Tests
# ============================================================================


class TestPresetParsing:
    def test_basic_symmetric(self):
        p = parse_tq_preset("tq_k8v8")
        assert p.k_bits == 8
        assert p.v_bits == 8
        assert not p.qjl

    def test_qjl_flag(self):
        p = parse_tq_preset("tq_k4v4_qjl")
        assert p.k_bits == 4
        assert p.v_bits == 4
        assert p.qjl

    def test_asymmetric(self):
        p = parse_tq_preset("tq_k4v8")
        assert p.k_bits == 4
        assert p.v_bits == 8

    def test_backward_compat_alias(self):
        p = parse_tq_preset("turboquant")
        assert p.k_bits == 8
        assert p.v_bits == 8
        assert not p.qjl  # MSE-only by default (QJL hurts attention)

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_tq_preset("fp8")
        with pytest.raises(ValueError):
            parse_tq_preset("tq_invalid")

    def test_bits_out_of_range(self):
        with pytest.raises(ValueError, match="k_bits must be 2-8"):
            parse_tq_preset("tq_k1v4")
        with pytest.raises(ValueError, match="v_bits must be 2-8"):
            parse_tq_preset("tq_k4v9")

    def test_is_tq_preset(self):
        assert is_tq_preset("tq_k8v8")
        assert is_tq_preset("tq_k4v4_qjl")
        assert is_tq_preset("turboquant")  # alias
        assert not is_tq_preset("fp8")
        assert not is_tq_preset("auto")


class TestPresetProperties:
    def test_byte_mode(self):
        p = parse_tq_preset("tq_k8v8")
        assert p.k_byte_mode
        assert p.v_byte_mode

    def test_nibble_mode(self):
        p = parse_tq_preset("tq_k4v4")
        assert not p.k_byte_mode
        assert not p.v_byte_mode

    def test_mixed_mode(self):
        p = parse_tq_preset("tq_k4v8")
        assert not p.k_byte_mode
        assert p.v_byte_mode

    def test_avg_bits_symmetric(self):
        p = parse_tq_preset("tq_k4v4")
        assert p.avg_bits_per_dim == 4.0

    def test_avg_bits_asymmetric(self):
        p = parse_tq_preset("tq_k4v8")
        assert p.avg_bits_per_dim == 6.0

    def test_cache_dim_byte_no_qjl(self):
        p = parse_tq_preset("tq_k8v8")
        # 128 bytes indices + 4 bytes norm = 132
        assert p.cache_dim_per_head(128, "k") == 132

    def test_cache_dim_nibble_no_qjl(self):
        p = parse_tq_preset("tq_k4v4")
        # 64 bytes packed indices + 4 bytes norm = 68
        assert p.cache_dim_per_head(128, "k") == 68

    def test_cache_dim_byte_qjl(self):
        p = parse_tq_preset("tq_k8v8_qjl")
        # qjl_padded_dim(128) = 128 + 4 + 16 + 4 = 152
        dim = p.cache_dim_per_head(128, "k")
        assert dim == 152

    def test_padded_cache_dim_aligned(self):
        p = parse_tq_preset("tq_k8v8_qjl")
        padded = p.padded_cache_dim(128)
        # 152 → aligned to 4 = 152 (already 4-byte aligned)
        assert padded == 152
        assert padded % 4 == 0


# ============================================================================
# Backend Class Tests
# ============================================================================


class TestBackendClass:
    def test_name(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )

        assert TurboQuantAttentionBackend.get_name() == "TURBOQUANT"

    def test_supports_kv_cache_dtype(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )

        assert TurboQuantAttentionBackend.supports_kv_cache_dtype("tq_k8v8_qjl")
        assert TurboQuantAttentionBackend.supports_kv_cache_dtype("tq_k4v4")
        assert TurboQuantAttentionBackend.supports_kv_cache_dtype("turboquant")
        assert not TurboQuantAttentionBackend.supports_kv_cache_dtype("fp8")
        assert not TurboQuantAttentionBackend.supports_kv_cache_dtype(None)

    def test_supports_head_size(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )

        assert TurboQuantAttentionBackend.supports_head_size(128)
        assert TurboQuantAttentionBackend.supports_head_size(64)
        assert TurboQuantAttentionBackend.supports_head_size(256)
        assert not TurboQuantAttentionBackend.supports_head_size(96)  # not power-of-2
        assert not TurboQuantAttentionBackend.supports_head_size(16)  # too small

    def test_cache_shape(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )

        shape = TurboQuantAttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            cache_dtype_str="tq_k8v8_qjl",
        )
        assert shape[0] == 100  # num_blocks
        assert shape[1] == 2  # K/V split
        assert shape[2] == 16  # block_size
        assert shape[3] == 4  # num_kv_heads
        assert shape[4] == 152  # padded_dim (152, already 4-byte aligned)

    def test_forward_excludes_kv_update(self):
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )

        assert not TurboQuantAttentionBackend.forward_includes_kv_cache_update


# ============================================================================
# Backend Registration Tests
# ============================================================================


class TestBackendRegistration:
    def test_enum_exists(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        assert hasattr(AttentionBackendEnum, "TURBOQUANT")

    def test_enum_resolves(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        path = AttentionBackendEnum.TURBOQUANT.get_path()
        assert "turboquant_attn.TurboQuantAttentionBackend" in path


# ============================================================================
# KV Quant Mode Tests
# ============================================================================


class TestKVQuantMode:
    def test_tq_byte_mode(self):
        from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

        mode = get_kv_quant_mode("tq_k8v8")
        assert mode == KVQuantMode.TURBOQUANT_BYTE

    def test_tq_nibble_mode(self):
        from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

        mode = get_kv_quant_mode("tq_k4v4")
        assert mode == KVQuantMode.TURBOQUANT

    def test_tq_mixed_mode(self):
        from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

        # k=8 (byte), v=4 (nibble) → MIXED (byte K + nibble V)
        mode = get_kv_quant_mode("tq_k8v4")
        assert mode == KVQuantMode.TURBOQUANT_MIXED

    def test_alias_mode(self):
        from vllm.v1.kv_cache_interface import get_kv_quant_mode

        mode = get_kv_quant_mode("turboquant")
        assert mode.is_turboquant


class TestBoundarySkipLayers:
    """Test auto-computed boundary layer protection."""

    def test_8bit_no_skip(self):
        from vllm.v1.attention.backends.turboquant_config import (
            get_boundary_skip_layers,
        )

        assert get_boundary_skip_layers(28, "tq_k8v8") == []

    def test_mixed_skip_first_last(self):
        from vllm.v1.attention.backends.turboquant_config import (
            get_boundary_skip_layers,
        )

        result = get_boundary_skip_layers(28, "tq_k8fv4")
        assert result == ["0", "27"]

    def test_4bit_skip_two_each(self):
        from vllm.v1.attention.backends.turboquant_config import (
            get_boundary_skip_layers,
        )

        result = get_boundary_skip_layers(28, "tq_k4v4")
        assert result == ["0", "1", "26", "27"]

    def test_small_model(self):
        from vllm.v1.attention.backends.turboquant_config import (
            get_boundary_skip_layers,
        )

        # 4-layer model: first 2 + last 2 overlap
        result = get_boundary_skip_layers(4, "tq_k4v4")
        assert result == ["0", "1", "2", "3"]

    def test_32_layer_model(self):
        from vllm.v1.attention.backends.turboquant_config import (
            get_boundary_skip_layers,
        )

        result = get_boundary_skip_layers(32, "tq_k4v4")
        assert result == ["0", "1", "30", "31"]

    def test_3bit_key_skip(self):
        """3-bit keys + 4-bit values: avg 3.5 bits ≤ 4 → skip first 2 + last 2."""
        from vllm.v1.attention.backends.turboquant_config import (
            get_boundary_skip_layers,
        )

        result = get_boundary_skip_layers(28, "tq_k3v4")
        assert result == ["0", "1", "26", "27"]


# ============================================================================
# 3-bit Preset Tests
# ============================================================================


class TestThreeBitPreset:
    """Tests for tq_k3v4 and other 3-bit configurations."""

    def test_parse_k3v4(self):
        p = parse_tq_preset("tq_k3v4")
        assert p.k_bits == 3
        assert p.v_bits == 4
        assert not p.qjl
        assert not p.k_fp8

    def test_parse_k3v3(self):
        p = parse_tq_preset("tq_k3v3")
        assert p.k_bits == 3
        assert p.v_bits == 3

    def test_3bit_nibble_mode(self):
        """3-bit keys use nibble mode (≤4 bit)."""
        p = parse_tq_preset("tq_k3v4")
        assert not p.k_byte_mode
        assert not p.v_byte_mode

    def test_3bit_avg_bits(self):
        p = parse_tq_preset("tq_k3v4")
        assert p.avg_bits_per_dim == 3.5

    def test_3bit_cache_dim(self):
        """3-bit still uses nibble packing (4-bit containers)."""
        p = parse_tq_preset("tq_k3v4")
        # Nibble mode: head_dim // 2 + 4 bytes norm = 64 + 4 = 68
        assert p.cache_dim_per_head(128, "k") == 68

    def test_3bit_kv_quant_mode(self):
        from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

        mode = get_kv_quant_mode("tq_k3v4")
        assert mode == KVQuantMode.TURBOQUANT

    def test_backward_compat_hyphen(self):
        p = parse_tq_preset("tq-k3v4")
        assert p.k_bits == 3
        assert p.v_bits == 4


# ============================================================================
# CacheDType Validation Tests
# ============================================================================


class TestCacheDTypeValidation:
    """Tests for dynamic TQ preset validation in CacheConfig."""

    def test_known_static_dtypes(self):
        """Static dtypes should be accepted."""
        from vllm.config.cache import CacheConfig

        for dtype in ["auto", "fp8", "bfloat16"]:
            cfg = CacheConfig(cache_dtype=dtype)
            assert cfg.cache_dtype == dtype

    def test_tq_preset_accepted(self):
        """All valid tq_* presets should be accepted."""
        from vllm.config.cache import CacheConfig

        for dtype in [
            "tq_k8v8",
            "tq_k4v4",
            "tq_k8fv4",
            "tq_k3v4",
            "tq_k8v8_qjl",
            "tq_k4v4_qjl",
            "turboquant",
        ]:
            cfg = CacheConfig(cache_dtype=dtype)
            assert cfg.cache_dtype == dtype

    def test_invalid_dtype_rejected(self):
        """Unknown dtypes should raise ValueError."""
        from pydantic import ValidationError

        from vllm.config.cache import CacheConfig

        with pytest.raises((ValueError, ValidationError)):
            CacheConfig(cache_dtype="not_a_dtype")

    def test_invalid_tq_format_rejected(self):
        """Malformed tq_ strings should raise ValueError."""
        from pydantic import ValidationError

        from vllm.config.cache import CacheConfig

        with pytest.raises((ValueError, ValidationError)):
            CacheConfig(cache_dtype="tq_invalid")
