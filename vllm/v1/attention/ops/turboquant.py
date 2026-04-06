# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant KV Cache Compression

Implements PolarQuant from "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate" (Google Research, ICLR 2026, arXiv:2504.19874).

Algorithm (matching the paper exactly):
  1. Random orthogonal rotation R via QR decomposition
  2. Normalize K/V to unit vectors, store L2 norm separately
  3. Rotate: y = R @ x_hat (coordinates become ~independent Beta((d-1)/2))
  4. Lloyd-Max scalar quantize each coordinate for Beta(d/2) on [-1,1]
  5. At attention time: rotate Q by R (Q_rot = Q @ R^T), no inverse
     rotation needed for K. Inverse-rotate output by R^T for V.

Key identity: q · k_recon = ||k|| · (R @ q)^T · centroids[idx_k]

QJL variant (Algorithm 2): adds 1-bit sign correction on quantization
residual.  For each coordinate, stores sign(y - centroid[idx]) as a packed
bit and mean(|residual|) as a per-token-head float32 scalar.  At attention
time the correction is:
  K_corrected = centroids[idx]*norm + res_scale * sign_vec * norm
"""

import math

import torch

from vllm.triton_utils import tl, triton

# ============================================================================
# QJL Utilities
# ============================================================================


def sign_bytes_padded(head_dim: int) -> int:
    """Number of bytes to store packed sign bits, 4-byte aligned.

    Each coordinate contributes 1 sign bit, packed 8 per byte.
    Padded to a multiple of 4 so the residual-norm float32 that
    follows is naturally aligned.
    """
    raw = head_dim // 8
    return (raw + 3) & ~3  # round up to next multiple of 4


def qjl_padded_dim(head_dim: int) -> int:
    """Total padded cache dimension per head for byte-mode + QJL.

    Layout: [head_dim indices | 4B L2 norm | sign_bytes | 4B res_scale]
    """
    return head_dim + 4 + sign_bytes_padded(head_dim) + 4


# ============================================================================
# Codebook Computation (CPU, one-time)
# ============================================================================


def _compute_exact_centroids(
    d: int,
    n_bits: int,
    n_iters: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Lloyd-Max centroids via numerical integration on Beta((d-1)/2).

    Uses scipy.integrate.quad for exact centroid computation instead of
    Monte Carlo sampling. This eliminates codebook noise from finite samples.

    Falls back to Monte Carlo if scipy is unavailable.
    """
    import numpy as np
    from scipy import integrate
    from scipy.stats import beta as beta_dist

    n_levels = 2**n_bits
    alpha = (d - 1) / 2.0

    # Beta(alpha, alpha) PDF on [-1, 1], mapped from standard [0, 1]
    def pdf(x: float) -> float:
        return float(beta_dist.pdf((x + 1) / 2, alpha, alpha) / 2)

    # Initialize centroids at distribution quantiles
    quantiles = np.linspace(0.5 / n_levels, 1 - 0.5 / n_levels, n_levels)
    centroids = np.array(beta_dist.ppf(quantiles, alpha, alpha) * 2 - 1)

    for _ in range(n_iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo = -1.0 if i == 0 else boundaries[i - 1]
            hi = 1.0 if i == n_levels - 1 else boundaries[i]
            num, _ = integrate.quad(lambda x: x * pdf(x), lo, hi)
            den, _ = integrate.quad(pdf, lo, hi)
            new_centroids[i] = num / den if den > 1e-15 else centroids[i]
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return (
        torch.tensor(boundaries, dtype=torch.float32),
        torch.tensor(centroids, dtype=torch.float32),
    )


def _compute_mc_centroids(
    d: int,
    n_bits: int,
    n_iters: int = 100,
    n_samples: int = 200000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte Carlo Lloyd-Max centroids for Beta((d-1)/2) on [-1, 1].

    Fallback when scipy is unavailable. Uses 200K samples for centroid
    estimation via vectorized scatter_add.
    """
    n_levels = 2**n_bits
    alpha = (d - 1) / 2.0

    beta_dist = torch.distributions.Beta(alpha, alpha)
    samples = beta_dist.sample((n_samples,)).double() * 2 - 1

    quantiles = torch.linspace(
        0.5 / n_levels, 1.0 - 0.5 / n_levels, n_levels, dtype=torch.float64
    )
    sorted_samples, _ = samples.sort()
    q_idx = (quantiles * n_samples).long().clamp(0, n_samples - 1)
    centroids = sorted_samples[q_idx]

    for _ in range(n_iters):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        assignments = torch.searchsorted(boundaries, samples)

        sums = torch.zeros(n_levels, dtype=torch.float64)
        counts = torch.zeros(n_levels, dtype=torch.float64)
        sums.scatter_add_(0, assignments, samples)
        counts.scatter_add_(0, assignments, torch.ones_like(samples))

        nonempty = counts > 0
        new_centroids = centroids.clone()
        new_centroids[nonempty] = sums[nonempty] / counts[nonempty]
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return boundaries.float(), centroids.float()


def compute_beta_centroids(
    d: int,
    n_bits: int,
    n_iters: int = 100,
    n_samples: int = 200000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd-Max centroids for Beta((d-1)/2) on [-1, 1].

    This is the distribution of a single coordinate of a uniformly random
    unit vector in R^d after orthogonal rotation.  For large d the
    distribution is tightly concentrated near zero (std ~ 1/sqrt(d)), so
    centroid initialization must place all levels inside the data range.

    Uses exact numerical integration (scipy) when available, falling back
    to Monte Carlo estimation with n_samples samples.

    Results are cached by (d, n_bits) since they're deterministic.

    Returns:
        boundaries: (2^n_bits - 1,) decision boundaries
        centroids: (2^n_bits,) reconstruction centroids
    """
    cache_key = (d, n_bits)
    if cache_key in _centroid_cache:
        return _centroid_cache[cache_key]
    try:
        result = _compute_exact_centroids(d, n_bits, n_iters)
    except ImportError:
        result = _compute_mc_centroids(d, n_bits, n_iters, n_samples)
    _centroid_cache[cache_key] = result
    return result


# Module-level cache for computed centroids (keyed by (d, n_bits))
_centroid_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _hadamard_matrix(d: int) -> torch.Tensor:
    """Walsh-Hadamard matrix of size d (must be power of 2)."""
    H = torch.tensor([[1.0]], device="cpu")
    k = 1
    while k < d:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
        k *= 2
    return H[:d, :d] / math.sqrt(d)


def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """Orthogonal rotation matrix for TurboQuant.

    For power-of-2 dimensions: randomized Hadamard (D @ H) — optimal for
    spreading energy uniformly across coordinates.

    For non-power-of-2 dimensions (e.g., outlier subgroups with d=96):
    random orthogonal via QR decomposition. Truncating a Hadamard matrix
    to non-power-of-2 size destroys orthogonality.

    Returns:
        R: (d, d) orthogonal matrix where R @ R^T = I
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    signs = (
        torch.randint(
            0, 2, (d,), generator=gen, device="cpu", dtype=torch.float32
        )
        * 2
        - 1
    )

    if d > 0 and (d & (d - 1)) == 0:
        # Power of 2: use randomized Hadamard
        H = _hadamard_matrix(d)
        return signs.unsqueeze(1) * H
    else:
        # Non-power-of-2: block-diagonal randomized Hadamard.
        # Decompose d into power-of-2 blocks (greedy: largest first).
        # Each block gets its own Hadamard rotation, ensuring orthogonality
        # and uniform energy spreading within each block.
        R = torch.zeros(d, d, dtype=torch.float32, device="cpu")
        remaining = d
        offset = 0
        while remaining > 0:
            block_size = 1 << (remaining.bit_length() - 1)  # largest pow2 <= remaining
            H_block = _hadamard_matrix(block_size)
            block_signs = signs[offset : offset + block_size]
            R[offset : offset + block_size, offset : offset + block_size] = (
                block_signs.unsqueeze(1) * H_block
            )
            offset += block_size
            remaining -= block_size
        return R


# ============================================================================
# TurboQuant Codebook
# ============================================================================


class TurboQuantCodebook:
    """Pre-computed TurboQuant codebook matching the paper exactly.

    When ``qjl=True``, also stores per-token-head sign bits and residual
    scale for a 1-bit correction on the quantization residual. This is a
    practical optimization (not the paper's Algorithm 2 which uses a random
    S matrix — that adds variance that hurts attention quality). Instead we
    store ``sign(residual)`` and ``mean(|residual|)`` directly, which
    empirically improves quality on real models (tested on Qwen2.5-7B).
    """

    def __init__(
        self,
        n_bits: int = 4,
        head_dim: int = 128,
        seed: int = 42,
        device: str = "cuda",
        qjl: bool = True,
    ):
        if n_bits > 8:
            raise ValueError(f"TurboQuant supports n_bits <= 8, got {n_bits}.")
        self.n_bits = n_bits
        self.n_levels = 2**n_bits
        self.head_dim = head_dim
        self.seed = seed
        self.byte_mode = n_bits > 4
        self.qjl = qjl
        self.sign_bytes = sign_bytes_padded(head_dim) if self.qjl else 0

        # Beta((d-1)/2) centroids on [-1, 1] (NOT N(0,1)!)
        boundaries, centroids = compute_beta_centroids(head_dim, n_bits)
        self.boundaries = boundaries.contiguous().to(device)
        self.centroids = centroids.contiguous().to(device)

        # Full orthogonal rotation (NOT sign flips!)
        R = generate_rotation_matrix(head_dim, seed)
        self.rotation_matrix = R.contiguous().to(device)
        # Pre-transposed for fast matmul in encode/rotate
        self.rotation_matrix_T = R.T.contiguous().to(device)

    def to(self, device) -> "TurboQuantCodebook":
        self.boundaries = self.boundaries.to(device)
        self.centroids = self.centroids.to(device)
        self.rotation_matrix = self.rotation_matrix.to(device)
        self.rotation_matrix_T = self.rotation_matrix_T.to(device)
        return self


# ============================================================================
# Nibble Packing (4-bit)
# ============================================================================


def pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices into nibble pairs.

    Convention: contiguous halves (NOT interleaved).
    packed[i] = indices[i] | (indices[i + d//2] << 4)

    Args:
        indices: (..., head_dim) uint8 with values in [0, 15]
    Returns:
        (..., head_dim // 2) uint8
    """
    d = indices.shape[-1]
    lo = indices[..., : d // 2]
    hi = indices[..., d // 2 :]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Unpack nibble pairs to 4-bit indices.

    Args:
        packed: (..., head_dim // 2) uint8
        head_dim: original head dimension
    Returns:
        (..., head_dim) uint8
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.cat([lo, hi], dim=-1)


# ============================================================================
# 2-bit Packing
# ============================================================================


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices: 4 values per byte.

    Convention: contiguous quarters.
    packed[i] = idx[4i] | (idx[4i+1] << 2) | (idx[4i+2] << 4) | (idx[4i+3] << 6)

    Args:
        indices: (..., dim) uint8 with values in [0, 3]
    Returns:
        (..., dim // 4) uint8
    """
    d = indices.shape[-1]
    assert d % 4 == 0
    reshaped = indices.reshape(*indices.shape[:-1], d // 4, 4)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=indices.device)
    packed = (reshaped << shifts).sum(dim=-1).to(torch.uint8)
    return packed


def unpack_2bit(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack 2-bit packed bytes to indices.

    Args:
        packed: (..., dim // 4) uint8
        dim: original dimension
    Returns:
        (..., dim) uint8 with values in [0, 3]
    """
    n_bytes = packed.shape[-1]
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & 0x03).reshape(
        *packed.shape[:-1], n_bytes * 4
    )
    return unpacked[..., :dim]


# ============================================================================
# Outlier Channel Handling (mixed-precision per channel group)
# ============================================================================


class OutlierChannelConfig:
    """Configuration for mixed-precision outlier channel quantization.

    Splits head_dim channels into outlier (high-variance) and regular
    groups. Each group gets its own bit width, rotation matrix, and
    centroids. This is the paper's approach for <4-bit quality.

    Example: head_dim=128, outlier_ratio=0.25, outlier_bits=4, regular_bits=2
    → 32 outlier channels at 4-bit + 96 regular channels at 2-bit
    → average 2.5 bits per channel
    """

    def __init__(
        self,
        head_dim: int = 128,
        outlier_ratio: float = 0.25,
        outlier_bits: int = 4,
        regular_bits: int = 2,
        seed: int = 42,
        qjl: bool = False,
        device: str = "cpu",
        outlier_mask: torch.Tensor | None = None,
    ):
        self.head_dim = head_dim
        self.outlier_ratio = outlier_ratio
        self.outlier_bits = outlier_bits
        self.regular_bits = regular_bits
        self.seed = seed
        self.qjl = qjl

        # Channel mask: True for outlier channels
        if outlier_mask is not None:
            self.outlier_mask = outlier_mask.to(device)
        else:
            # Default: first outlier_dim channels are outlier
            # (overridden by calibration)
            outlier_dim = int(head_dim * outlier_ratio)
            self.outlier_mask = torch.zeros(head_dim, dtype=torch.bool, device=device)
            self.outlier_mask[:outlier_dim] = True

        self.outlier_idx = self.outlier_mask.nonzero(as_tuple=True)[0]
        self.regular_idx = (~self.outlier_mask).nonzero(as_tuple=True)[0]
        self.outlier_dim = self.outlier_idx.shape[0]
        self.regular_dim = self.regular_idx.shape[0]

        avg_bits = (
            self.outlier_dim * outlier_bits + self.regular_dim * regular_bits
        ) / head_dim
        self.avg_bits = avg_bits

        # Separate codebooks for each group
        self.outlier_cb = TurboQuantCodebook(
            n_bits=outlier_bits,
            head_dim=self.outlier_dim,
            seed=seed,
            device=device,
            qjl=qjl,
        )
        self.regular_cb = TurboQuantCodebook(
            n_bits=regular_bits,
            head_dim=self.regular_dim,
            seed=seed + 1,  # different rotation
            device=device,
            qjl=qjl,
        )

    def to(self, device) -> "OutlierChannelConfig":
        self.outlier_mask = self.outlier_mask.to(device)
        self.outlier_idx = self.outlier_idx.to(device)
        self.regular_idx = self.regular_idx.to(device)
        self.outlier_cb = self.outlier_cb.to(device)
        self.regular_cb = self.regular_cb.to(device)
        return self

    @staticmethod
    def from_calibration(
        calibration_path: str,
        layer_idx: int,
        outlier_bits: int = 4,
        regular_bits: int = 2,
        device: str = "cpu",
        qjl: bool = False,
    ) -> "OutlierChannelConfig":
        """Load per-layer config from a calibration file.

        Args:
            calibration_path: Path to .pt file from turboquant_calibrate.
            layer_idx: Which attention layer.
        """
        cal = torch.load(calibration_path, map_location="cpu", weights_only=True)
        mask = cal["masks"][layer_idx]
        head_dim = cal["head_dim"]
        outlier_ratio = cal["outlier_ratio"]
        return OutlierChannelConfig(
            head_dim=head_dim,
            outlier_ratio=outlier_ratio,
            outlier_bits=outlier_bits,
            regular_bits=regular_bits,
            seed=42,
            qjl=qjl,
            device=device,
            outlier_mask=mask,
        )

    def cache_bytes_per_head(self) -> int:
        """Total bytes per token-head in the cache."""
        # Outlier: nibble-packed if 4-bit, byte if >4
        if self.outlier_bits <= 4:
            out_bytes = self.outlier_dim // 2
        else:
            out_bytes = self.outlier_dim
        # Regular: 2-bit packed (4 per byte) if 2-bit, nibble if 3-4
        if self.regular_bits <= 2:
            reg_bytes = (self.regular_dim + 3) // 4
        elif self.regular_bits <= 4:
            reg_bytes = self.regular_dim // 2
        else:
            reg_bytes = self.regular_dim
        norm_bytes = 8  # 2 × float32 (outlier_norm + regular_norm)
        return out_bytes + reg_bytes + norm_bytes


def calibrate_outlier_channels(
    model_or_activations: torch.Tensor,
    outlier_ratio: float = 0.25,
) -> torch.Tensor:
    """Identify outlier channels from calibration activations.

    Args:
        model_or_activations: [num_tokens, num_kv_heads, head_dim] tensor
            of K or V activations from a calibration run.
        outlier_ratio: Fraction of channels to mark as outlier.

    Returns:
        outlier_mask: [head_dim] boolean tensor, True for outlier channels.
    """
    # Compute per-channel variance across tokens and heads
    if model_or_activations.dim() == 3:
        # [tokens, heads, dim] → variance over tokens and heads
        var = model_or_activations.float().var(dim=(0, 1))
    elif model_or_activations.dim() == 2:
        var = model_or_activations.float().var(dim=0)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {model_or_activations.dim()}D")

    head_dim = var.shape[0]
    n_outlier = int(head_dim * outlier_ratio)
    _, top_idx = var.topk(n_outlier)
    mask = torch.zeros(head_dim, dtype=torch.bool, device=var.device)
    mask[top_idx] = True
    return mask


def outlier_encode_ref(
    tensor: torch.Tensor,
    config: OutlierChannelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference encode with outlier channel splitting.

    Each subgroup (outlier, regular) is quantized independently with its
    own L2 norm, rotation matrix, and codebook. Per-subgroup norms are
    critical: subsets of a unit vector in R^d are NOT unit vectors in
    their subspaces.

    Args:
        tensor: (..., head_dim) input tensor

    Returns:
        outlier_indices: (..., outlier_dim) uint8
        regular_indices: (..., regular_dim) uint8
        outlier_norms: (...) float32 — L2 norms of outlier subgroup
        regular_norms: (...) float32 — L2 norms of regular subgroup
    """
    device = tensor.device
    config_dev = config.to(device)

    # Split channels BEFORE normalization
    x_out = tensor.float()[..., config_dev.outlier_idx]  # (..., outlier_dim)
    x_reg = tensor.float()[..., config_dev.regular_idx]  # (..., regular_dim)

    # Per-subgroup L2 norms (NOT shared!)
    norm_out = x_out.norm(dim=-1)  # (...)
    norm_reg = x_reg.norm(dim=-1)  # (...)

    # Normalize each subgroup to unit sphere in its own subspace
    x_out_hat = x_out / (norm_out.unsqueeze(-1) + 1e-10)
    x_reg_hat = x_reg / (norm_reg.unsqueeze(-1) + 1e-10)

    # Rotate each group independently
    R_out_T = config_dev.outlier_cb.rotation_matrix_T
    R_reg_T = config_dev.regular_cb.rotation_matrix_T
    shape_out = x_out_hat.shape
    shape_reg = x_reg_hat.shape
    y_out = (x_out_hat.reshape(-1, shape_out[-1]) @ R_out_T).reshape(shape_out)
    y_reg = (x_reg_hat.reshape(-1, shape_reg[-1]) @ R_reg_T).reshape(shape_reg)

    # Quantize each group
    out_cb = config_dev.outlier_cb
    reg_cb = config_dev.regular_cb

    out_idx = torch.zeros_like(y_out, dtype=torch.uint8)
    for i in range(out_cb.n_levels - 1):
        out_idx += (y_out > out_cb.boundaries[i]).to(torch.uint8)

    reg_idx = torch.zeros_like(y_reg, dtype=torch.uint8)
    for i in range(reg_cb.n_levels - 1):
        reg_idx += (y_reg > reg_cb.boundaries[i]).to(torch.uint8)

    # Norm correction per subgroup
    out_centroid_vals = out_cb.centroids[out_idx.long()]
    out_cnorm = torch.norm(out_centroid_vals.float(), dim=-1)
    norm_out_corrected = norm_out / (out_cnorm + 1e-12)

    reg_centroid_vals = reg_cb.centroids[reg_idx.long()]
    reg_cnorm = torch.norm(reg_centroid_vals.float(), dim=-1)
    norm_reg_corrected = norm_reg / (reg_cnorm + 1e-12)

    return out_idx, reg_idx, norm_out_corrected, norm_reg_corrected


def outlier_decode_ref(
    outlier_indices: torch.Tensor,
    regular_indices: torch.Tensor,
    outlier_norms: torch.Tensor,
    regular_norms: torch.Tensor,
    config: OutlierChannelConfig,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference decode with outlier channel reconstruction.

    Reconstructs in ORIGINAL space (applies inverse rotation per group,
    scales by per-subgroup norms, scatters to original positions).
    """
    device = outlier_indices.device
    config_dev = config.to(device)

    # Centroid lookup per group
    out_vals = config_dev.outlier_cb.centroids[outlier_indices.long()]
    reg_vals = config_dev.regular_cb.centroids[regular_indices.long()]

    # Inverse-rotate each group to original space
    R_out = config_dev.outlier_cb.rotation_matrix
    R_reg = config_dev.regular_cb.rotation_matrix
    shape_out = out_vals.shape
    shape_reg = reg_vals.shape
    x_out = (out_vals.float().reshape(-1, shape_out[-1]) @ R_out).reshape(shape_out)
    x_reg = (reg_vals.float().reshape(-1, shape_reg[-1]) @ R_reg).reshape(shape_reg)

    # Scale by per-subgroup norms (NOT shared)
    x_out = outlier_norms.unsqueeze(-1) * x_out
    x_reg = regular_norms.unsqueeze(-1) * x_reg

    # Scatter back to original positions
    head_dim = config_dev.head_dim
    output_shape = (*outlier_norms.shape, head_dim)
    output = torch.zeros(output_shape, dtype=torch.float32, device=device)
    output[..., config_dev.outlier_idx] = x_out
    output[..., config_dev.regular_idx] = x_reg

    return output.to(output_dtype)


def outlier_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_out_norms: torch.Tensor,
    k_reg_norms: torch.Tensor,
    v_out_norms: torch.Tensor,
    v_reg_norms: torch.Tensor,
    slot_mapping: torch.Tensor,
    config: OutlierChannelConfig,
) -> None:
    """Encode K/V with outlier channel splitting into paged cache.

    Each subgroup (outlier, regular) is independently normalized, rotated,
    and quantized with its own codebook and per-subgroup L2 norm.

    Storage layout per token-head:
    [outlier_packed | regular_packed]
    Norms stored separately in strided float32 views.

    For 4-bit outlier + 2-bit regular with d=128:
    [16B nibble | 24B 2bit-packed] = 40 bytes indices
    + 4B outlier_norm + 4B regular_norm = 48 bytes total
    """
    valid = slot_mapping >= 0
    valid_slots = slot_mapping[valid]
    dev = key.device
    config_dev = config.to(dev)
    block_size = key_cache.shape[1]

    for src, cache, out_norms, reg_norms in [
        (key, key_cache, k_out_norms, k_reg_norms),
        (value, value_cache, v_out_norms, v_reg_norms),
    ]:
        x = src[valid].float()  # (n_valid, nkv, head_dim)

        # Split channels BEFORE normalization
        x_out = x[..., config_dev.outlier_idx]
        x_reg = x[..., config_dev.regular_idx]

        # Per-subgroup L2 norms
        nrm_out = x_out.norm(dim=-1)  # (n_valid, nkv)
        nrm_reg = x_reg.norm(dim=-1)

        # Normalize each subgroup independently
        x_out_hat = x_out / (nrm_out.unsqueeze(-1) + 1e-10)
        x_reg_hat = x_reg / (nrm_reg.unsqueeze(-1) + 1e-10)

        # Rotate each group
        R_out_T = config_dev.outlier_cb.rotation_matrix_T
        R_reg_T = config_dev.regular_cb.rotation_matrix_T
        s_out = x_out_hat.shape
        s_reg = x_reg_hat.shape
        y_out = (x_out_hat.reshape(-1, s_out[-1]) @ R_out_T).reshape(s_out)
        y_reg = (x_reg_hat.reshape(-1, s_reg[-1]) @ R_reg_T).reshape(s_reg)

        # Quantize outlier
        out_cb = config_dev.outlier_cb
        out_idx = torch.zeros_like(y_out, dtype=torch.uint8)
        for i in range(out_cb.n_levels - 1):
            out_idx += (y_out > out_cb.boundaries[i]).to(torch.uint8)

        # Norm correction for outlier
        out_centroid_vals = out_cb.centroids[out_idx.long()]
        out_cnorm = torch.norm(out_centroid_vals.float(), dim=-1)
        nrm_out_corrected = nrm_out / (out_cnorm + 1e-12)

        # Quantize regular
        reg_cb = config_dev.regular_cb
        reg_idx = torch.zeros_like(y_reg, dtype=torch.uint8)
        for i in range(reg_cb.n_levels - 1):
            reg_idx += (y_reg > reg_cb.boundaries[i]).to(torch.uint8)

        # Norm correction for regular
        reg_centroid_vals = reg_cb.centroids[reg_idx.long()]
        reg_cnorm = torch.norm(reg_centroid_vals.float(), dim=-1)
        nrm_reg_corrected = nrm_reg / (reg_cnorm + 1e-12)

        # Pack indices
        out_packed = pack_nibbles(out_idx)  # (..., outlier_dim//2)
        reg_packed = pack_2bit(reg_idx)  # (..., regular_dim//4)

        # Scatter to paged cache
        n_valid = x.shape[0]
        nkv = x.shape[1]
        out_bytes = out_packed.shape[-1]
        reg_bytes = reg_packed.shape[-1]

        for tok_i in range(n_valid):
            slot = valid_slots[tok_i].item()
            blk = slot // block_size
            off = slot % block_size
            for h in range(nkv):
                cache[blk, off, h, :out_bytes] = out_packed[tok_i, h]
                cache[blk, off, h, out_bytes:out_bytes + reg_bytes] = (
                    reg_packed[tok_i, h]
                )
                out_norms[blk, off, h] = nrm_out_corrected[tok_i, h]
                reg_norms[blk, off, h] = nrm_reg_corrected[tok_i, h]


def outlier_dequant_paged(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_out_norms: torch.Tensor,
    k_reg_norms: torch.Tensor,
    v_out_norms: torch.Tensor,
    v_reg_norms: torch.Tensor,
    staging_key: torch.Tensor,
    staging_value: torch.Tensor,
    config: OutlierChannelConfig,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks_per_seq: int,
) -> None:
    """Decompress outlier-mode TQ paged cache to bf16 staging buffer.

    Uses per-subgroup L2 norms (outlier_norm and regular_norm stored
    separately). Outputs in ORIGINAL space (inverse-rotates per group,
    scatters back). No Q rotation or output rotation needed after this.
    """
    num_seqs = block_table.shape[0]
    if num_seqs == 0:
        return

    dev = key_cache.device
    config_dev = config.to(dev)
    block_size = key_cache.shape[1]
    nkv = key_cache.shape[2]
    head_dim = config_dev.head_dim
    out_dim = config_dev.outlier_dim
    reg_dim = config_dev.regular_dim
    out_bytes = out_dim // 2  # nibble packed
    reg_bytes = (reg_dim + 3) // 4  # 2-bit packed

    R_out = config_dev.outlier_cb.rotation_matrix
    R_reg = config_dev.regular_cb.rotation_matrix
    out_centroids = config_dev.outlier_cb.centroids
    reg_centroids = config_dev.regular_cb.centroids

    for src_cache, src_out_norms, src_reg_norms, staging in [
        (key_cache, k_out_norms, k_reg_norms, staging_key),
        (value_cache, v_out_norms, v_reg_norms, staging_value),
    ]:
        for s in range(num_seqs):
            seq_len = seq_lens[s].item()
            n_blocks = (seq_len + block_size - 1) // block_size
            for p in range(n_blocks):
                phys_blk = block_table[s, p].item()
                staging_blk = s * max_blocks_per_seq + p
                for slot in range(block_size):
                    for h in range(nkv):
                        norm_out = src_out_norms[phys_blk, slot, h].item()
                        norm_reg = src_reg_norms[phys_blk, slot, h].item()
                        if norm_out == 0.0 and norm_reg == 0.0:
                            staging[staging_blk, slot, h, :] = 0.0
                            continue

                        # Read outlier packed nibbles
                        out_raw = src_cache[phys_blk, slot, h, :out_bytes]
                        out_idx = unpack_nibbles(
                            out_raw.unsqueeze(0), out_dim
                        ).squeeze(0)
                        out_vals = out_centroids[out_idx.long()]

                        # Read regular 2-bit packed
                        reg_raw = src_cache[
                            phys_blk, slot, h, out_bytes:out_bytes + reg_bytes
                        ]
                        reg_idx = unpack_2bit(
                            reg_raw.unsqueeze(0), reg_dim
                        ).squeeze(0)
                        reg_vals = reg_centroids[reg_idx.long()]

                        # Inverse-rotate to original space
                        x_out = (out_vals.float() @ R_out).to(torch.float32)
                        x_reg = (reg_vals.float() @ R_reg).to(torch.float32)

                        # Scale by per-subgroup norms and scatter
                        full = torch.zeros(head_dim, device=dev)
                        full[config_dev.outlier_idx] = norm_out * x_out
                        full[config_dev.regular_idx] = norm_reg * x_reg
                        staging[staging_blk, slot, h] = full.to(torch.bfloat16)


# ============================================================================
# Triton Encode Kernel (fused quantize + pack + scatter)
# ============================================================================


@triton.jit
def _turboquant_encode_packed_kernel(
    # Rotated + normalized input
    rotated_ptr,
    stride_r_tok: tl.int64,
    stride_r_head: tl.int64,
    stride_r_dim: tl.int64,
    # Packed cache output
    cache_ptr,
    stride_c_blk: tl.int64,
    stride_c_slot: tl.int64,
    stride_c_head: tl.int64,
    stride_c_dim: tl.int64,
    # Norms output (strided float32 view into cache padding)
    norms_ptr,
    stride_n_blk: tl.int64,
    stride_n_slot: tl.int64,
    stride_n_head: tl.int64,
    # Pre-computed norms
    orig_norms_ptr,
    stride_on_tok: tl.int64,
    stride_on_head: tl.int64,
    # Slot mapping
    slot_mapping_ptr,
    # Quantization boundaries + centroids
    boundaries_ptr,
    centroids_ptr,
    # QJL sign output (uint8 packed bits in cache)
    signs_ptr,
    stride_sg_blk: tl.int64,
    stride_sg_slot: tl.int64,
    stride_sg_head: tl.int64,
    stride_sg_byte: tl.int64,
    # QJL residual scale output (float32 strided view)
    res_scale_ptr,
    stride_rs_blk: tl.int64,
    stride_rs_slot: tl.int64,
    stride_rs_head: tl.int64,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_LEVELS: tl.constexpr,
    HALF_DIM: tl.constexpr,
    QJL_ENABLED: tl.constexpr,
    SIGN_BYTES_HALF: tl.constexpr,
    LOG2_LEVELS: tl.constexpr = 8,
):
    """Fused quantize + nibble-pack + scatter into paged cache."""
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE

    # Load pre-computed L2 norm
    norm = tl.load(orig_norms_ptr + tok * stride_on_tok + head * stride_on_head)

    # Load both halves of rotated normalized vector
    base = tok * stride_r_tok + head * stride_r_head
    offs_lo = tl.arange(0, HALF_DIM)
    offs_hi = HALF_DIM + tl.arange(0, HALF_DIM)

    y_lo = tl.load(rotated_ptr + base + offs_lo * stride_r_dim).to(tl.float32)
    y_hi = tl.load(rotated_ptr + base + offs_hi * stride_r_dim).to(tl.float32)

    # Vectorized binary search: find quantization level for each coordinate.
    # idx = number of boundaries where y > boundary (= bisect_left position).
    # O(log2(N_LEVELS)) iterations instead of O(N_LEVELS).
    n_boundaries: tl.constexpr = N_LEVELS - 1
    lo_lo = tl.zeros([HALF_DIM], dtype=tl.int32)
    hi_lo = tl.full([HALF_DIM], n_boundaries, dtype=tl.int32)
    lo_hi = tl.zeros([HALF_DIM], dtype=tl.int32)
    hi_hi = tl.full([HALF_DIM], n_boundaries, dtype=tl.int32)
    for _ in range(LOG2_LEVELS):
        mid_lo = (lo_lo + hi_lo) // 2
        mid_hi = (lo_hi + hi_hi) // 2
        b_lo = tl.load(boundaries_ptr + mid_lo)
        b_hi = tl.load(boundaries_ptr + mid_hi)
        gt_lo = y_lo > b_lo
        gt_hi = y_hi > b_hi
        lo_lo = tl.where(gt_lo, mid_lo + 1, lo_lo)
        hi_lo = tl.where(gt_lo, hi_lo, mid_lo)
        lo_hi = tl.where(gt_hi, mid_hi + 1, lo_hi)
        hi_hi = tl.where(gt_hi, hi_hi, mid_hi)
    idx_lo = lo_lo.to(tl.uint8)
    idx_hi = lo_hi.to(tl.uint8)

    # Pack nibbles: low half → low nibble, high half → high nibble
    packed = idx_lo | (idx_hi << 4)

    # Scatter packed indices to cache
    c_base = blk * stride_c_blk + off * stride_c_slot + head * stride_c_head
    tl.store(cache_ptr + c_base + offs_lo * stride_c_dim, packed)

    # Look up centroids for each half (needed for norm correction and QJL)
    c_lo = tl.load(centroids_ptr + idx_lo.to(tl.int32))
    c_hi = tl.load(centroids_ptr + idx_hi.to(tl.int32))

    # Norm correction: centroid vector ||c[idx]|| != 1 after coordinate-wise
    # quantization. Dividing by ||c[idx]|| ensures the reconstructed vector
    # has the correct magnitude.
    centroid_norm_sq = tl.sum(c_lo * c_lo, axis=0) + tl.sum(c_hi * c_hi, axis=0)
    centroid_norm = tl.sqrt(centroid_norm_sq + 1e-12)
    corrected_norm = norm / centroid_norm

    # Scatter corrected norm (float32 via strided view)
    n_off = blk * stride_n_blk + off * stride_n_slot + head * stride_n_head
    tl.store(norms_ptr + n_off, corrected_norm)

    # --- QJL: compute residual per half, pack signs, store res_scale ---
    if QJL_ENABLED:
        r_lo = y_lo - c_lo
        r_hi = y_hi - c_hi

        # res_scale = mean(|residual|) over full HEAD_DIM
        res_scale = (
            tl.sum(tl.abs(r_lo), axis=0) + tl.sum(tl.abs(r_hi), axis=0)
        ) / HEAD_DIM
        rs_off = blk * stride_rs_blk + off * stride_rs_slot + head * stride_rs_head
        tl.store(res_scale_ptr + rs_off, res_scale)

        # Pack lo-half sign bits (coords 0..HALF_DIM-1 → bytes 0..SIGN_BYTES_HALF-1)
        sign_lo = (r_lo >= 0).to(tl.int32)
        bit_lo = (offs_lo % 8).to(tl.int32)
        shifted_lo = sign_lo << bit_lo
        packed_lo_2d = tl.reshape(shifted_lo, [SIGN_BYTES_HALF, 8])
        packed_lo_bytes = tl.sum(packed_lo_2d, axis=1).to(tl.uint8)

        # Pack hi-half sign bits
        # coords HALF_DIM..HEAD_DIM-1 → bytes SIGN_BYTES_HALF..2*SBH-1
        sign_hi = (r_hi >= 0).to(tl.int32)
        shifted_hi = sign_hi << bit_lo  # same bit positions
        packed_hi_2d = tl.reshape(shifted_hi, [SIGN_BYTES_HALF, 8])
        packed_hi_bytes = tl.sum(packed_hi_2d, axis=1).to(tl.uint8)

        # Store sign bytes
        sg_base = blk * stride_sg_blk + off * stride_sg_slot + head * stride_sg_head
        offs_sb = tl.arange(0, SIGN_BYTES_HALF)
        tl.store(signs_ptr + sg_base + offs_sb * stride_sg_byte, packed_lo_bytes)
        tl.store(
            signs_ptr + sg_base + (SIGN_BYTES_HALF + offs_sb) * stride_sg_byte,
            packed_hi_bytes,
        )


@triton.jit
def _turboquant_encode_byte_kernel(
    # Rotated + normalized input
    rotated_ptr,
    stride_r_tok: tl.int64,
    stride_r_head: tl.int64,
    stride_r_dim: tl.int64,
    # Byte cache output (1 uint8 per coordinate)
    cache_ptr,
    stride_c_blk: tl.int64,
    stride_c_slot: tl.int64,
    stride_c_head: tl.int64,
    stride_c_dim: tl.int64,
    # Norms output (strided float32 view into cache padding)
    norms_ptr,
    stride_n_blk: tl.int64,
    stride_n_slot: tl.int64,
    stride_n_head: tl.int64,
    # Pre-computed norms
    orig_norms_ptr,
    stride_on_tok: tl.int64,
    stride_on_head: tl.int64,
    # Slot mapping
    slot_mapping_ptr,
    # Quantization boundaries + centroids
    boundaries_ptr,
    centroids_ptr,
    # QJL sign output (uint8 packed bits in cache)
    signs_ptr,  # cache ptr offset to sign region
    stride_sg_blk: tl.int64,
    stride_sg_slot: tl.int64,
    stride_sg_head: tl.int64,
    stride_sg_byte: tl.int64,
    # QJL residual scale output (float32 strided view)
    res_scale_ptr,
    stride_rs_blk: tl.int64,
    stride_rs_slot: tl.int64,
    stride_rs_head: tl.int64,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_LEVELS: tl.constexpr,
    QJL_ENABLED: tl.constexpr,
    SIGN_BYTES: tl.constexpr,
    LOG2_LEVELS: tl.constexpr = 8,
):
    """Fused quantize + scatter into paged cache (byte storage, 5-8 bit).

    When QJL_ENABLED, also stores packed sign bits and mean-absolute-residual
    for each token-head.
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE

    # Load pre-computed L2 norm
    norm = tl.load(orig_norms_ptr + tok * stride_on_tok + head * stride_on_head)

    # Load full rotated normalized vector
    base = tok * stride_r_tok + head * stride_r_head
    offs_d = tl.arange(0, HEAD_DIM)
    y = tl.load(rotated_ptr + base + offs_d * stride_r_dim).to(tl.float32)

    # Vectorized binary search for quantization level
    n_boundaries: tl.constexpr = N_LEVELS - 1
    lo = tl.zeros([HEAD_DIM], dtype=tl.int32)
    hi = tl.full([HEAD_DIM], n_boundaries, dtype=tl.int32)
    for _ in range(LOG2_LEVELS):
        mid = (lo + hi) // 2
        b_mid = tl.load(boundaries_ptr + mid)
        gt = y > b_mid
        lo = tl.where(gt, mid + 1, lo)
        hi = tl.where(gt, hi, mid)
    idx = lo.to(tl.uint8)

    # Scatter raw uint8 indices to cache (no packing)
    c_base = blk * stride_c_blk + off * stride_c_slot + head * stride_c_head
    tl.store(cache_ptr + c_base + offs_d * stride_c_dim, idx)

    # Look up centroids for quantized indices (needed for norm correction
    # and optionally for QJL residual computation)
    centroid_vals = tl.load(centroids_ptr + idx.to(tl.int32))

    # Norm correction: centroid vector ||c[idx]|| != 1 after coordinate-wise
    # quantization. Dividing by ||c[idx]|| ensures the reconstructed vector
    # K = c[idx] * corrected_norm has ||K|| == original ||K||.
    centroid_norm_sq = tl.sum(centroid_vals * centroid_vals, axis=0)
    centroid_norm = tl.sqrt(centroid_norm_sq + 1e-12)
    corrected_norm = norm / centroid_norm

    # Scatter corrected norm (float32 via strided view)
    n_off = blk * stride_n_blk + off * stride_n_slot + head * stride_n_head
    tl.store(norms_ptr + n_off, corrected_norm)

    # --- QJL: compute residual, pack signs, store res_scale ---
    if QJL_ENABLED:
        # Residual: y - centroid (reusing centroid_vals from above)
        residual = y - centroid_vals

        # res_scale = mean(|residual|) = sum(|r|) / d
        res_scale = tl.sum(tl.abs(residual), axis=0) / HEAD_DIM

        # Store res_scale (float32 via strided view)
        rs_off = blk * stride_rs_blk + off * stride_rs_slot + head * stride_rs_head
        tl.store(res_scale_ptr + rs_off, res_scale)

        # Pack sign bits: sign_byte[j] = OR over bits 0..7 of
        #   ((residual[j*8+bit] >= 0) << bit)
        # Using reshape + sum trick: since bits are non-overlapping,
        # sum == OR for these shifted single-bit values.
        sign_raw = (residual >= 0).to(tl.int32)  # (HEAD_DIM,) 0/1
        bit_pos = (offs_d % 8).to(tl.int32)  # (HEAD_DIM,) 0-7
        shifted = sign_raw << bit_pos  # (HEAD_DIM,)
        # Reshape to (SIGN_BYTES, 8) — works because HEAD_DIM may have
        # padding but SIGN_BYTES * 8 >= HEAD_DIM.  We zero-pad to
        # SIGN_BYTES * 8 via the initial zeros above.
        packed_2d = tl.reshape(shifted, [SIGN_BYTES, 8])
        packed = tl.sum(packed_2d, axis=1).to(tl.uint8)  # (SIGN_BYTES,)

        # Scatter packed sign bytes
        sg_base = blk * stride_sg_blk + off * stride_sg_slot + head * stride_sg_head
        offs_sb = tl.arange(0, SIGN_BYTES)
        tl.store(signs_ptr + sg_base + offs_sb * stride_sg_byte, packed)


# ============================================================================
# Outlier Encode Kernel (Triton: quantize + mixed pack + scatter)
# ============================================================================


@triton.jit
def _outlier_encode_kernel(
    # Rotated outlier subgroup (after split, normalize, rotate in PyTorch)
    out_rotated_ptr,
    stride_or_tok: tl.int64,
    stride_or_head: tl.int64,
    stride_or_dim: tl.int64,
    # Rotated regular subgroup
    reg_rotated_ptr,
    stride_rr_tok: tl.int64,
    stride_rr_head: tl.int64,
    stride_rr_dim: tl.int64,
    # Packed cache output
    cache_ptr,
    stride_c_blk: tl.int64,
    stride_c_slot: tl.int64,
    stride_c_head: tl.int64,
    stride_c_dim: tl.int64,
    # Outlier norms output (float32 strided view)
    out_norms_ptr,
    stride_on_blk: tl.int64,
    stride_on_slot: tl.int64,
    stride_on_head: tl.int64,
    # Regular norms output (float32 strided view)
    reg_norms_ptr,
    stride_rn_blk: tl.int64,
    stride_rn_slot: tl.int64,
    stride_rn_head: tl.int64,
    # Pre-computed per-subgroup norms
    orig_out_norms_ptr,
    stride_oon_tok: tl.int64,
    stride_oon_head: tl.int64,
    orig_reg_norms_ptr,
    stride_orn_tok: tl.int64,
    stride_orn_head: tl.int64,
    # Slot mapping
    slot_mapping_ptr,
    # Outlier quantization params
    out_boundaries_ptr,
    out_centroids_ptr,
    # Regular quantization params
    reg_boundaries_ptr,
    reg_centroids_ptr,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    OUTLIER_DIM: tl.constexpr,
    REGULAR_DIM: tl.constexpr,
    OUT_HALF_DIM: tl.constexpr,
    REG_QUARTER_DIM: tl.constexpr,
    OUT_N_LEVELS: tl.constexpr,
    REG_N_LEVELS: tl.constexpr,
    OUT_LOG2: tl.constexpr,
    REG_LOG2: tl.constexpr,
    OUT_BYTES: tl.constexpr,
    REG_BYTES: tl.constexpr,
    # Padded dimensions (next power of 2) for tl.arange
    REGULAR_DIM_PAD: tl.constexpr = 128,
    REG_QUARTER_DIM_PAD: tl.constexpr = 32,
):
    """Fused outlier encode: quantize both groups + nibble/2bit pack + scatter.

    Grid: (num_tokens, num_heads).
    Each program encodes one token-head pair for both outlier and regular groups.
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE

    # --- Outlier group: 4-bit nibble-packed ---
    out_norm = tl.load(
        orig_out_norms_ptr + tok * stride_oon_tok + head * stride_oon_head
    )

    out_base = tok * stride_or_tok + head * stride_or_head
    offs_out_half = tl.arange(0, OUT_HALF_DIM)
    offs_out_lo = offs_out_half
    offs_out_hi = OUT_HALF_DIM + offs_out_half

    y_out_lo = tl.load(
        out_rotated_ptr + out_base + offs_out_lo * stride_or_dim
    ).to(tl.float32)
    y_out_hi = tl.load(
        out_rotated_ptr + out_base + offs_out_hi * stride_or_dim
    ).to(tl.float32)

    # Binary search for outlier quantization levels
    out_n_boundaries: tl.constexpr = OUT_N_LEVELS - 1
    lo_lo = tl.zeros([OUT_HALF_DIM], dtype=tl.int32)
    hi_lo = tl.full([OUT_HALF_DIM], out_n_boundaries, dtype=tl.int32)
    lo_hi = tl.zeros([OUT_HALF_DIM], dtype=tl.int32)
    hi_hi = tl.full([OUT_HALF_DIM], out_n_boundaries, dtype=tl.int32)
    for _ in range(OUT_LOG2):
        mid_lo = (lo_lo + hi_lo) // 2
        mid_hi = (lo_hi + hi_hi) // 2
        b_lo = tl.load(out_boundaries_ptr + mid_lo)
        b_hi = tl.load(out_boundaries_ptr + mid_hi)
        lo_lo = tl.where(y_out_lo > b_lo, mid_lo + 1, lo_lo)
        hi_lo = tl.where(y_out_lo > b_lo, hi_lo, mid_lo)
        lo_hi = tl.where(y_out_hi > b_hi, mid_hi + 1, lo_hi)
        hi_hi = tl.where(y_out_hi > b_hi, hi_hi, mid_hi)
    idx_out_lo = lo_lo.to(tl.uint8)
    idx_out_hi = lo_hi.to(tl.uint8)

    # Nibble pack: lo | (hi << 4)
    packed_out = idx_out_lo | (idx_out_hi << 4)

    # Norm correction for outlier subgroup
    c_out_lo = tl.load(out_centroids_ptr + idx_out_lo.to(tl.int32))
    c_out_hi = tl.load(out_centroids_ptr + idx_out_hi.to(tl.int32))
    out_cn_sq = tl.sum(c_out_lo * c_out_lo, axis=0) + tl.sum(
        c_out_hi * c_out_hi, axis=0
    )
    out_cn = tl.sqrt(out_cn_sq + 1e-12)
    corrected_out_norm = out_norm / out_cn

    # Scatter outlier packed nibbles to cache
    c_base = blk * stride_c_blk + off * stride_c_slot + head * stride_c_head
    tl.store(cache_ptr + c_base + offs_out_half * stride_c_dim, packed_out)

    # Store corrected outlier norm
    on_off = blk * stride_on_blk + off * stride_on_slot + head * stride_on_head
    tl.store(out_norms_ptr + on_off, corrected_out_norm)

    # --- Regular group: 2-bit packed (4 per byte) ---
    # Triton requires tl.arange sizes to be powers of 2.
    # Pad to REGULAR_DIM_PAD and mask out-of-bounds.
    reg_norm = tl.load(
        orig_reg_norms_ptr + tok * stride_orn_tok + head * stride_orn_head
    )

    reg_base = tok * stride_rr_tok + head * stride_rr_head
    offs_reg = tl.arange(0, REGULAR_DIM_PAD)
    reg_mask = offs_reg < REGULAR_DIM
    y_reg = tl.load(
        reg_rotated_ptr + reg_base + offs_reg * stride_rr_dim,
        mask=reg_mask, other=0.0,
    ).to(tl.float32)

    # Binary search for regular quantization levels
    reg_n_boundaries: tl.constexpr = REG_N_LEVELS - 1
    lo_r = tl.zeros([REGULAR_DIM_PAD], dtype=tl.int32)
    hi_r = tl.full([REGULAR_DIM_PAD], reg_n_boundaries, dtype=tl.int32)
    for _ in range(REG_LOG2):
        mid_r = (lo_r + hi_r) // 2
        b_r = tl.load(reg_boundaries_ptr + mid_r)
        lo_r = tl.where(y_reg > b_r, mid_r + 1, lo_r)
        hi_r = tl.where(y_reg > b_r, hi_r, mid_r)
    idx_reg = tl.where(reg_mask, lo_r, 0).to(tl.uint8)

    # 2-bit pack: 4 indices per byte via shift + sum
    # Use padded REG_QUARTER_DIM_PAD for reshape, mask when storing
    idx_reg_i32 = idx_reg.to(tl.int32)
    # Trim to REGULAR_DIM, pad remainder to 0 for clean packing
    idx_trimmed = tl.where(offs_reg < REGULAR_DIM, idx_reg_i32, 0)
    # Reshape to (REG_QUARTER_DIM_PAD, 4)
    idx_reg_2d = tl.reshape(idx_trimmed, [REG_QUARTER_DIM_PAD, 4])
    shifts = tl.arange(0, 4) * 2  # [0, 2, 4, 6]
    packed_reg = tl.sum(idx_reg_2d << shifts[None, :], axis=1).to(tl.uint8)

    # Norm correction for regular subgroup (only valid dims)
    c_reg = tl.load(reg_centroids_ptr + idx_reg.to(tl.int32))
    c_reg_masked = tl.where(reg_mask, c_reg, 0.0)
    reg_cn_sq = tl.sum(c_reg_masked * c_reg_masked, axis=0)
    reg_cn = tl.sqrt(reg_cn_sq + 1e-12)
    corrected_reg_norm = reg_norm / reg_cn

    # Scatter regular packed 2-bit bytes to cache (after outlier region)
    reg_store_offs = tl.arange(0, REG_QUARTER_DIM_PAD)
    reg_store_mask = reg_store_offs < REG_QUARTER_DIM
    tl.store(
        cache_ptr + c_base + (OUT_BYTES + reg_store_offs) * stride_c_dim,
        packed_reg,
        mask=reg_store_mask,
    )

    # Store corrected regular norm
    rn_off = blk * stride_rn_blk + off * stride_rn_slot + head * stride_rn_head
    tl.store(reg_norms_ptr + rn_off, corrected_reg_norm)


def outlier_encode_single(
    tensor: torch.Tensor,
    cache: torch.Tensor,
    out_norms_cache: torch.Tensor,
    reg_norms_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    config: "OutlierChannelConfig",
) -> None:
    """Encode a single tensor (K or V) with outlier channel splitting.

    PyTorch handles channel split, per-group normalization, and rotation
    (cuBLAS matmul). Triton handles quantization, packing, and scatter.

    CUDAGraph-safe: no boolean indexing, no dynamic shapes. All tokens
    processed; kernel skips slots < 0.
    """
    num_tokens = tensor.shape[0]
    if num_tokens == 0:
        return

    dev = tensor.device
    config_dev = config.to(dev)
    n_heads = tensor.shape[1]

    # Split channels (static indices — CUDAGraph safe)
    x_out = tensor.float()[..., config_dev.outlier_idx]  # (T, H, out_dim)
    x_reg = tensor.float()[..., config_dev.regular_idx]  # (T, H, reg_dim)

    # Per-subgroup L2 norms
    nrm_out = x_out.norm(dim=-1)  # (T, H)
    nrm_reg = x_reg.norm(dim=-1)  # (T, H)

    # Normalize each subgroup to unit sphere
    x_out_hat = x_out / (nrm_out.unsqueeze(-1) + 1e-10)
    x_reg_hat = x_reg / (nrm_reg.unsqueeze(-1) + 1e-10)

    # Rotate each subgroup (cuBLAS matmul — fast)
    out_R_T = config_dev.outlier_cb.rotation_matrix_T
    reg_R_T = config_dev.regular_cb.rotation_matrix_T
    s_out = x_out_hat.shape
    s_reg = x_reg_hat.shape
    y_out = (x_out_hat.reshape(-1, s_out[-1]) @ out_R_T).reshape(s_out)
    y_reg = (x_reg_hat.reshape(-1, s_reg[-1]) @ reg_R_T).reshape(s_reg)

    # Triton: quantize + pack + scatter
    out_cb = config_dev.outlier_cb
    reg_cb = config_dev.regular_cb
    out_log2 = math.ceil(math.log2(max(out_cb.n_levels, 2)))
    reg_log2 = math.ceil(math.log2(max(reg_cb.n_levels, 2)))
    out_half = config_dev.outlier_dim // 2
    reg_quarter = (config_dev.regular_dim + 3) // 4
    out_bytes = out_half  # nibble-packed = half the coordinates

    grid = (num_tokens, n_heads)
    _outlier_encode_kernel[grid](
        y_out, y_out.stride(0), y_out.stride(1), y_out.stride(2),
        y_reg, y_reg.stride(0), y_reg.stride(1), y_reg.stride(2),
        cache, cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
        out_norms_cache,
        out_norms_cache.stride(0), out_norms_cache.stride(1), out_norms_cache.stride(2),
        reg_norms_cache,
        reg_norms_cache.stride(0), reg_norms_cache.stride(1), reg_norms_cache.stride(2),
        nrm_out, nrm_out.stride(0), nrm_out.stride(1),
        nrm_reg, nrm_reg.stride(0), nrm_reg.stride(1),
        slot_mapping,
        out_cb.boundaries, out_cb.centroids,
        reg_cb.boundaries, reg_cb.centroids,
        BLOCK_SIZE=cache.shape[1],
        OUTLIER_DIM=config_dev.outlier_dim,
        REGULAR_DIM=config_dev.regular_dim,
        OUT_HALF_DIM=out_half,
        REG_QUARTER_DIM=reg_quarter,
        OUT_N_LEVELS=out_cb.n_levels,
        REG_N_LEVELS=reg_cb.n_levels,
        OUT_LOG2=out_log2,
        REG_LOG2=reg_log2,
        OUT_BYTES=out_bytes,
        REG_BYTES=reg_quarter,
        REGULAR_DIM_PAD=triton.next_power_of_2(config_dev.regular_dim),
        REG_QUARTER_DIM_PAD=triton.next_power_of_2(reg_quarter),
    )


# ============================================================================
# Outlier Unpack Kernel (Triton: unpack + centroid lookup + norm scale)
# ============================================================================


@triton.jit
def _outlier_unpack_kernel(
    # TQ cache (packed indices)
    cache_ptr,
    stride_c_blk: tl.int64,
    stride_c_slot: tl.int64,
    stride_c_head: tl.int64,
    stride_c_dim: tl.int64,
    # Dual norms (float32 strided views)
    out_norms_ptr,
    stride_on_blk: tl.int64,
    stride_on_slot: tl.int64,
    stride_on_head: tl.int64,
    reg_norms_ptr,
    stride_rn_blk: tl.int64,
    stride_rn_slot: tl.int64,
    stride_rn_head: tl.int64,
    # Centroids
    out_centroids_ptr,
    reg_centroids_ptr,
    # Output: rotated-space centroid values * norm
    out_result_ptr,
    stride_or_blk: tl.int64,
    stride_or_slot: tl.int64,
    stride_or_head: tl.int64,
    stride_or_dim: tl.int64,
    reg_result_ptr,
    stride_rr_blk: tl.int64,
    stride_rr_slot: tl.int64,
    stride_rr_head: tl.int64,
    stride_rr_dim: tl.int64,
    # Block table
    block_table_ptr,
    stride_bt_seq: tl.int64,
    stride_bt_pos: tl.int64,
    seq_lens_ptr,
    # Grid info
    max_blocks_per_seq: tl.int64,
    num_seqs: tl.int64,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    OUTLIER_DIM: tl.constexpr,
    REGULAR_DIM: tl.constexpr,
    OUT_HALF_DIM: tl.constexpr,
    REG_QUARTER_DIM: tl.constexpr,
    OUT_BYTES: tl.constexpr,
    REG_QUARTER_DIM_PAD: tl.constexpr = 32,
):
    """Unpack outlier cache: read packed indices, lookup centroids, scale by norms.

    Grid: (num_seqs * max_blocks_per_seq, num_kv_heads).
    Each program handles one (seq_block, head), looping over BLOCK_SIZE slots.
    Output is in ROTATED space (caller does inverse rotation via cuBLAS).
    """
    prog_id = tl.program_id(0)
    head = tl.program_id(1)

    seq_idx = prog_id // max_blocks_per_seq
    block_pos = prog_id % max_blocks_per_seq

    if seq_idx >= num_seqs:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if block_pos >= n_blocks:
        return

    phys_blk = tl.load(
        block_table_ptr + seq_idx * stride_bt_seq + block_pos * stride_bt_pos
    ).to(tl.int64)
    staging_blk = seq_idx * max_blocks_per_seq + block_pos

    for slot in range(BLOCK_SIZE):
        # Load dual norms
        on_off = phys_blk * stride_on_blk + slot * stride_on_slot + head * stride_on_head
        rn_off = phys_blk * stride_rn_blk + slot * stride_rn_slot + head * stride_rn_head
        norm_out = tl.load(out_norms_ptr + on_off)
        norm_reg = tl.load(reg_norms_ptr + rn_off)

        # --- Outlier: unpack nibbles → centroid lookup → scale ---
        c_base = (
            phys_blk * stride_c_blk
            + slot * stride_c_slot
            + head * stride_c_head
        )
        offs_half = tl.arange(0, OUT_HALF_DIM)
        packed = tl.load(
            cache_ptr + c_base + offs_half * stride_c_dim
        )
        idx_lo = (packed & 0x0F).to(tl.int32)
        idx_hi = ((packed >> 4) & 0x0F).to(tl.int32)
        c_lo = tl.load(out_centroids_ptr + idx_lo)
        c_hi = tl.load(out_centroids_ptr + idx_hi)

        # Write outlier centroid values * norm to output buffer
        o_base = (
            staging_blk * stride_or_blk
            + slot * stride_or_slot
            + head * stride_or_head
        )
        tl.store(
            out_result_ptr + o_base + offs_half * stride_or_dim,
            (norm_out * c_lo).to(tl.float32),
        )
        tl.store(
            out_result_ptr + o_base + (OUT_HALF_DIM + offs_half) * stride_or_dim,
            (norm_out * c_hi).to(tl.float32),
        )

        # --- Regular: unpack 2-bit → centroid lookup → scale ---
        reg_byte_offs = tl.arange(0, REG_QUARTER_DIM_PAD)
        reg_byte_mask = reg_byte_offs < REG_QUARTER_DIM
        reg_packed = tl.load(
            cache_ptr + c_base + (OUT_BYTES + reg_byte_offs) * stride_c_dim,
            mask=reg_byte_mask, other=0,
        )

        # Unpack 4 indices per byte
        r_base = (
            staging_blk * stride_rr_blk
            + slot * stride_rr_slot
            + head * stride_rr_head
        )
        for bit_pair in range(4):
            shift = bit_pair * 2
            idx_r = ((reg_packed.to(tl.int32) >> shift) & 0x03)
            c_r = tl.load(reg_centroids_ptr + idx_r)
            dim_offs = reg_byte_offs * 4 + bit_pair
            # Only store if within REGULAR_DIM
            mask = dim_offs < REGULAR_DIM
            tl.store(
                reg_result_ptr + r_base + dim_offs * stride_rr_dim,
                (norm_reg * c_r).to(tl.float32),
                mask=mask,
            )


def outlier_dequant_to_staging(
    cache: torch.Tensor,
    out_norms: torch.Tensor,
    reg_norms: torch.Tensor,
    staging: torch.Tensor,
    config: "OutlierChannelConfig",
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks_per_seq: int,
    out_rotated_buf: torch.Tensor,
    reg_rotated_buf: torch.Tensor,
) -> None:
    """Decompress outlier TQ cache to bf16 staging buffer.

    Three-step pipeline:
    1. Triton kernel: unpack indices, lookup centroids, scale by norms
       → writes to intermediate rotated-space buffers
    2. PyTorch (cuBLAS): inverse-rotate each subgroup
    3. PyTorch: scatter to staging buffer at original channel positions

    Args:
        cache: (num_blocks, block_size, nkv, padded_dim) uint8
        out_norms: (num_blocks, block_size, nkv) float32 strided view
        reg_norms: same
        staging: (max_staging_blocks, block_size, nkv, head_dim) bf16
        config: OutlierChannelConfig
        block_table: (num_seqs, max_blocks_per_seq) int32
        seq_lens: (num_seqs,) int32
        max_blocks_per_seq: int
        out_rotated_buf: (max_staging_blocks, block_size, nkv, outlier_dim) f32
        reg_rotated_buf: (max_staging_blocks, block_size, nkv, regular_dim) f32
    """
    num_seqs = block_table.shape[0]
    if num_seqs == 0:
        return

    dev = cache.device
    config_dev = config.to(dev)
    nkv = cache.shape[2]
    out_half = config_dev.outlier_dim // 2
    reg_quarter = (config_dev.regular_dim + 3) // 4

    # Step 1: Triton unpack + centroid lookup + norm scaling
    grid = (num_seqs * max_blocks_per_seq, nkv)
    _outlier_unpack_kernel[grid](
        cache,
        cache.stride(0), cache.stride(1), cache.stride(2), cache.stride(3),
        out_norms,
        out_norms.stride(0), out_norms.stride(1), out_norms.stride(2),
        reg_norms,
        reg_norms.stride(0), reg_norms.stride(1), reg_norms.stride(2),
        config_dev.outlier_cb.centroids,
        config_dev.regular_cb.centroids,
        out_rotated_buf,
        out_rotated_buf.stride(0), out_rotated_buf.stride(1),
        out_rotated_buf.stride(2), out_rotated_buf.stride(3),
        reg_rotated_buf,
        reg_rotated_buf.stride(0), reg_rotated_buf.stride(1),
        reg_rotated_buf.stride(2), reg_rotated_buf.stride(3),
        block_table,
        block_table.stride(0), block_table.stride(1),
        seq_lens,
        max_blocks_per_seq=max_blocks_per_seq,
        num_seqs=num_seqs,
        BLOCK_SIZE=cache.shape[1],
        OUTLIER_DIM=config_dev.outlier_dim,
        REGULAR_DIM=config_dev.regular_dim,
        OUT_HALF_DIM=out_half,
        REG_QUARTER_DIM=reg_quarter,
        OUT_BYTES=out_half,  # nibble-packed = half the coordinates
        REG_QUARTER_DIM_PAD=triton.next_power_of_2(reg_quarter),
    )

    # Step 2: Inverse rotation (cuBLAS matmul — fast, parallel)
    R_out = config_dev.outlier_cb.rotation_matrix  # (out_dim, out_dim)
    R_reg = config_dev.regular_cb.rotation_matrix  # (reg_dim, reg_dim)

    n_staging = num_seqs * max_blocks_per_seq
    block_size = cache.shape[1]
    out_dim = config_dev.outlier_dim
    reg_dim = config_dev.regular_dim

    # Reshape for batch matmul, inverse rotate, reshape back
    out_flat = out_rotated_buf[:n_staging].reshape(-1, out_dim)
    reg_flat = reg_rotated_buf[:n_staging].reshape(-1, reg_dim)
    out_orig = (out_flat @ R_out).reshape(n_staging, block_size, nkv, out_dim)
    reg_orig = (reg_flat @ R_reg).reshape(n_staging, block_size, nkv, reg_dim)

    # Step 3: Scatter to staging at original channel positions
    staging[:n_staging].zero_()
    staging[:n_staging, :, :, config_dev.outlier_idx] = out_orig.to(
        torch.bfloat16
    )
    staging[:n_staging, :, :, config_dev.regular_idx] = reg_orig.to(
        torch.bfloat16
    )


# ============================================================================
# Encode (PyTorch rotation + Triton quantize/pack/scatter)
# ============================================================================


def turboquant_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_norms: torch.Tensor,
    v_norms: torch.Tensor,
    slot_mapping: torch.Tensor,
    codebook: TurboQuantCodebook,
    k_signs: torch.Tensor | None = None,
    v_signs: torch.Tensor | None = None,
    k_res_scales: torch.Tensor | None = None,
    v_res_scales: torch.Tensor | None = None,
) -> None:
    """Encode K/V with TurboQuant and scatter into paged cache.

    Paper algorithm:
      1. norm = ||x||
      2. x_hat = x / norm (unit vector)
      3. y = R @ x_hat (rotate, coordinates in [-1,1])
      4. idx = quantize(y) using Beta(d/2) centroids
      5. Pack nibbles and store packed uint8 + float32 norm
    """
    num_tokens = key.shape[0]
    if num_tokens == 0:
        return

    valid = slot_mapping >= 0
    valid_slots = slot_mapping[valid]

    qjl = codebook.qjl
    R_T = codebook.rotation_matrix_T  # Pre-transposed
    items = [
        (key, key_cache, k_norms, k_signs, k_res_scales),
        (value, value_cache, v_norms, v_signs, v_res_scales),
    ]

    for src, cache, norms_cache, signs_cache, res_scales_cache in items:
        x = src[valid].float()  # (n_valid, num_heads, head_dim)

        # 1. Compute norms (PyTorch)
        nrm = x.norm(dim=-1)  # (n_valid, num_heads)

        # 2. Normalize to unit vectors (PyTorch)
        x_hat = x / (nrm.unsqueeze(-1) + 1e-10)

        # 3. Rotate: y = x_hat @ R^T (cuBLAS matmul — faster than einsum)
        orig_shape = x_hat.shape
        y = (x_hat.reshape(-1, orig_shape[-1]) @ R_T).reshape(orig_shape)

        # 4+5. Quantize + (optionally pack) + scatter (Triton kernel)
        # The kernel also computes sign(residual) and mean(|r|) when QJL enabled.
        n_valid = x.shape[0]
        n_heads = x.shape[1]
        log2_levels = math.ceil(math.log2(max(codebook.n_levels, 2)))
        if n_valid > 0:
            grid = (n_valid, n_heads)
            if codebook.byte_mode:
                _turboquant_encode_byte_kernel[grid](
                    y,
                    y.stride(0),
                    y.stride(1),
                    y.stride(2),
                    cache,
                    cache.stride(0),
                    cache.stride(1),
                    cache.stride(2),
                    cache.stride(3),
                    norms_cache,
                    norms_cache.stride(0),
                    norms_cache.stride(1),
                    norms_cache.stride(2),
                    nrm,
                    nrm.stride(0),
                    nrm.stride(1),
                    valid_slots,
                    codebook.boundaries,
                    codebook.centroids,
                    signs_cache if qjl else cache,
                    signs_cache.stride(0) if qjl else 0,
                    signs_cache.stride(1) if qjl else 0,
                    signs_cache.stride(2) if qjl else 0,
                    signs_cache.stride(3) if qjl else 0,
                    res_scales_cache if qjl else norms_cache,
                    res_scales_cache.stride(0) if qjl else 0,
                    res_scales_cache.stride(1) if qjl else 0,
                    res_scales_cache.stride(2) if qjl else 0,
                    BLOCK_SIZE=cache.shape[1],
                    HEAD_DIM=codebook.head_dim,
                    N_LEVELS=codebook.n_levels,
                    QJL_ENABLED=qjl,
                    SIGN_BYTES=(codebook.head_dim + 7) // 8 if qjl else 1,
                    LOG2_LEVELS=log2_levels,
                )
            else:
                hd = codebook.head_dim
                _turboquant_encode_packed_kernel[grid](
                    y,
                    y.stride(0),
                    y.stride(1),
                    y.stride(2),
                    cache,
                    cache.stride(0),
                    cache.stride(1),
                    cache.stride(2),
                    cache.stride(3),
                    norms_cache,
                    norms_cache.stride(0),
                    norms_cache.stride(1),
                    norms_cache.stride(2),
                    nrm,
                    nrm.stride(0),
                    nrm.stride(1),
                    valid_slots,
                    codebook.boundaries,
                    codebook.centroids,
                    signs_cache if qjl else cache,
                    signs_cache.stride(0) if qjl else 0,
                    signs_cache.stride(1) if qjl else 0,
                    signs_cache.stride(2) if qjl else 0,
                    signs_cache.stride(3) if qjl else 0,
                    res_scales_cache if qjl else norms_cache,
                    res_scales_cache.stride(0) if qjl else 0,
                    res_scales_cache.stride(1) if qjl else 0,
                    res_scales_cache.stride(2) if qjl else 0,
                    BLOCK_SIZE=cache.shape[1],
                    HEAD_DIM=hd,
                    N_LEVELS=codebook.n_levels,
                    HALF_DIM=hd // 2,
                    QJL_ENABLED=qjl,
                    SIGN_BYTES_HALF=(hd // 2 + 7) // 8 if qjl else 1,
                    LOG2_LEVELS=log2_levels,
                )


def rotate_query(query: torch.Tensor, R_T: torch.Tensor) -> torch.Tensor:
    """Rotate query by R: q_rot = q @ R^T (batch matmul).

    Uses query's dtype (bf16) directly — no float32 cast. Rotation
    matrices should be pre-cast to the model dtype at init time.
    """
    shape = query.shape
    d = shape[-1]
    R_T = R_T.to(query.dtype)
    return (query.reshape(-1, d) @ R_T).reshape(shape)


def inverse_rotate_output(
    output: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """Inverse-rotate attention output: out @ R (batch matmul).

    Needed because V is in rotated space. Uses output's dtype directly.
    """
    shape = output.shape
    d = shape[-1]
    R = R.to(output.dtype)
    return (output.reshape(-1, d) @ R).reshape(shape)


# ============================================================================
# Triton Dequant Kernels (dequant-first architecture)
# ============================================================================


@triton.jit
def _turboquant_dequant_byte_kernel(
    # TQ cache (uint8 indices)
    cache_ptr,
    stride_c_blk: tl.int64,
    stride_c_slot: tl.int64,
    stride_c_head: tl.int64,
    stride_c_dim: tl.int64,
    # Norms (float32 strided view)
    norms_ptr,
    stride_n_blk: tl.int64,
    stride_n_slot: tl.int64,
    stride_n_head: tl.int64,
    # Signs (uint8 strided view, QJL)
    signs_ptr,
    stride_sg_blk: tl.int64,
    stride_sg_slot: tl.int64,
    stride_sg_head: tl.int64,
    stride_sg_byte: tl.int64,
    # Residual scales (float32 strided view, QJL)
    res_scales_ptr,
    stride_rs_blk: tl.int64,
    stride_rs_slot: tl.int64,
    stride_rs_head: tl.int64,
    # Centroids
    centroids_ptr,
    # Block table
    block_table_ptr,
    stride_bt_seq: tl.int64,
    stride_bt_pos: tl.int64,
    # Seq lens
    seq_lens_ptr,
    # Output staging buffer (bf16)
    out_ptr,
    stride_o_blk: tl.int64,
    stride_o_slot: tl.int64,
    stride_o_head: tl.int64,
    stride_o_dim: tl.int64,
    # Grid info
    max_blocks_per_seq: tl.int64,
    num_seqs: tl.int64,
    # Dirty block flags (bool tensor, per physical block)
    dirty_blocks_ptr,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QJL_ENABLED: tl.constexpr,
    SIGN_BYTES: tl.constexpr,
    DIRTY_CHECK: tl.constexpr = False,
):
    """Decompress TQ byte-mode blocks to bf16 staging buffer.

    Grid: (num_seqs * max_blocks_per_seq, num_kv_heads)
    Each program decompresses one (seq, block_pos, head) — all BLOCK_SIZE
    slots. The inner loop is compile-time unrolled by Triton.

    When DIRTY_CHECK is True, skips blocks where dirty_blocks[phys_blk]
    is False (incremental dequant — staging already has correct data).
    """
    flat_id = tl.program_id(0)
    head = tl.program_id(1)

    seq_id = flat_id // max_blocks_per_seq
    block_pos = flat_id % max_blocks_per_seq

    if seq_id >= num_seqs:
        return
    seq_len = tl.load(seq_lens_ptr + seq_id)
    max_valid_block = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if block_pos >= max_valid_block:
        return

    phys_blk = tl.load(
        block_table_ptr + seq_id * stride_bt_seq + block_pos * stride_bt_pos
    ).to(tl.int64)

    # Incremental dequant: skip clean blocks (staging already valid)
    if DIRTY_CHECK:
        is_dirty = tl.load(dirty_blocks_ptr + phys_blk)
        if not is_dirty:
            return

    # Index staging by physical block (matches unified_attention's
    # block_table lookup into the staging buffer).
    staging_blk = phys_blk
    offs_d = tl.arange(0, HEAD_DIM)

    for slot in range(BLOCK_SIZE):
        n_off = phys_blk * stride_n_blk + slot * stride_n_slot + head * stride_n_head
        norm = tl.load(norms_ptr + n_off)

        o_base = (
            staging_blk * stride_o_blk + slot * stride_o_slot + head * stride_o_head
        )

        if norm == 0.0:
            tl.store(
                out_ptr + o_base + offs_d * stride_o_dim,
                tl.zeros([HEAD_DIM], dtype=tl.bfloat16),
            )
        else:
            c_base = (
                phys_blk * stride_c_blk + slot * stride_c_slot + head * stride_c_head
            )
            indices = tl.load(cache_ptr + c_base + offs_d * stride_c_dim)
            vals = tl.load(centroids_ptr + indices.to(tl.int32))

            if QJL_ENABLED:
                rs_off = (
                    phys_blk * stride_rs_blk
                    + slot * stride_rs_slot
                    + head * stride_rs_head
                )
                res_scale = tl.load(res_scales_ptr + rs_off)
                sg_base = (
                    phys_blk * stride_sg_blk
                    + slot * stride_sg_slot
                    + head * stride_sg_head
                )
                byte_idx = offs_d // 8
                bit_idx = (offs_d % 8).to(tl.int32)
                sign_byte_vals = tl.load(
                    signs_ptr + sg_base + byte_idx * stride_sg_byte
                )
                sign_bits = (sign_byte_vals.to(tl.int32) >> bit_idx) & 1
                sign_vec = 2.0 * sign_bits.to(tl.float32) - 1.0
                vals = vals + res_scale * sign_vec

            result = norm * vals
            tl.store(out_ptr + o_base + offs_d * stride_o_dim, result.to(tl.bfloat16))


@triton.jit
def _turboquant_dequant_nibble_kernel(
    # TQ cache (nibble-packed uint8 indices)
    cache_ptr,
    stride_c_blk: tl.int64,
    stride_c_slot: tl.int64,
    stride_c_head: tl.int64,
    stride_c_dim: tl.int64,
    # Norms (float32 strided view)
    norms_ptr,
    stride_n_blk: tl.int64,
    stride_n_slot: tl.int64,
    stride_n_head: tl.int64,
    # Signs (uint8 strided view, QJL)
    signs_ptr,
    stride_sg_blk: tl.int64,
    stride_sg_slot: tl.int64,
    stride_sg_head: tl.int64,
    stride_sg_byte: tl.int64,
    # Residual scales (float32 strided view, QJL)
    res_scales_ptr,
    stride_rs_blk: tl.int64,
    stride_rs_slot: tl.int64,
    stride_rs_head: tl.int64,
    # Centroids
    centroids_ptr,
    # Block table
    block_table_ptr,
    stride_bt_seq: tl.int64,
    stride_bt_pos: tl.int64,
    # Seq lens
    seq_lens_ptr,
    # Output staging buffer (bf16)
    out_ptr,
    stride_o_blk: tl.int64,
    stride_o_slot: tl.int64,
    stride_o_head: tl.int64,
    stride_o_dim: tl.int64,
    # Grid info
    max_blocks_per_seq: tl.int64,
    num_seqs: tl.int64,
    # Dirty block flags (bool tensor, per physical block)
    dirty_blocks_ptr,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    QJL_ENABLED: tl.constexpr,
    SIGN_BYTES_HALF: tl.constexpr,
    DIRTY_CHECK: tl.constexpr = False,
):
    """Decompress TQ nibble-mode blocks to bf16 staging buffer.

    Grid: (num_seqs * max_blocks_per_seq, num_kv_heads)
    Each program decompresses one (seq, block_pos, head) — all slots.
    """
    flat_id = tl.program_id(0)
    head = tl.program_id(1)

    seq_id = flat_id // max_blocks_per_seq
    block_pos = flat_id % max_blocks_per_seq

    if seq_id >= num_seqs:
        return
    seq_len = tl.load(seq_lens_ptr + seq_id)
    max_valid_block = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    if block_pos >= max_valid_block:
        return

    phys_blk = tl.load(
        block_table_ptr + seq_id * stride_bt_seq + block_pos * stride_bt_pos
    ).to(tl.int64)

    # Incremental dequant: skip clean blocks (staging already valid)
    if DIRTY_CHECK:
        is_dirty = tl.load(dirty_blocks_ptr + phys_blk)
        if not is_dirty:
            return

    # Index staging by physical block (matches unified_attention's
    # block_table lookup into the staging buffer).
    staging_blk = phys_blk
    offs_lo = tl.arange(0, HALF_DIM)
    offs_full_lo = tl.arange(0, HALF_DIM)
    offs_full_hi = HALF_DIM + tl.arange(0, HALF_DIM)

    for slot in range(BLOCK_SIZE):
        n_off = phys_blk * stride_n_blk + slot * stride_n_slot + head * stride_n_head
        norm = tl.load(norms_ptr + n_off)

        o_base = (
            staging_blk * stride_o_blk + slot * stride_o_slot + head * stride_o_head
        )

        if norm == 0.0:
            tl.store(
                out_ptr + o_base + offs_full_lo * stride_o_dim,
                tl.zeros([HALF_DIM], dtype=tl.bfloat16),
            )
            tl.store(
                out_ptr + o_base + offs_full_hi * stride_o_dim,
                tl.zeros([HALF_DIM], dtype=tl.bfloat16),
            )
        else:
            c_base = (
                phys_blk * stride_c_blk + slot * stride_c_slot + head * stride_c_head
            )
            packed = tl.load(cache_ptr + c_base + offs_lo * stride_c_dim)
            idx_lo = packed & 0x0F
            idx_hi = (packed >> 4) & 0x0F
            vals_lo = tl.load(centroids_ptr + idx_lo.to(tl.int32))
            vals_hi = tl.load(centroids_ptr + idx_hi.to(tl.int32))

            if QJL_ENABLED:
                rs_off = (
                    phys_blk * stride_rs_blk
                    + slot * stride_rs_slot
                    + head * stride_rs_head
                )
                res_scale = tl.load(res_scales_ptr + rs_off)
                sg_base = (
                    phys_blk * stride_sg_blk
                    + slot * stride_sg_slot
                    + head * stride_sg_head
                )
                byte_idx_lo = offs_lo // 8
                bit_idx_lo = (offs_lo % 8).to(tl.int32)
                sg_lo = tl.load(signs_ptr + sg_base + byte_idx_lo * stride_sg_byte)
                bits_lo = (sg_lo.to(tl.int32) >> bit_idx_lo) & 1
                sign_lo = 2.0 * bits_lo.to(tl.float32) - 1.0
                vals_lo = vals_lo + res_scale * sign_lo
                byte_idx_hi = offs_lo // 8
                bit_idx_hi = (offs_lo % 8).to(tl.int32)
                sg_hi = tl.load(
                    signs_ptr
                    + sg_base
                    + (SIGN_BYTES_HALF + byte_idx_hi) * stride_sg_byte
                )
                bits_hi = (sg_hi.to(tl.int32) >> bit_idx_hi) & 1
                sign_hi = 2.0 * bits_hi.to(tl.float32) - 1.0
                vals_hi = vals_hi + res_scale * sign_hi

            out_lo = norm * vals_lo
            out_hi = norm * vals_hi
            tl.store(
                out_ptr + o_base + offs_full_lo * stride_o_dim, out_lo.to(tl.bfloat16)
            )
            tl.store(
                out_ptr + o_base + offs_full_hi * stride_o_dim, out_hi.to(tl.bfloat16)
            )


def turboquant_dequant_paged(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_norms: torch.Tensor,
    v_norms: torch.Tensor,
    staging_key: torch.Tensor,
    staging_value: torch.Tensor,
    codebook: TurboQuantCodebook,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks_per_seq: int,
    k_signs: torch.Tensor | None = None,
    v_signs: torch.Tensor | None = None,
    k_res_scales: torch.Tensor | None = None,
    v_res_scales: torch.Tensor | None = None,
    dirty_blocks: torch.Tensor | None = None,
) -> None:
    """Decompress TQ paged cache blocks to bf16 staging buffers.

    Only decompresses blocks referenced by block_table. Data stays in
    rotated space (Q rotation and output inverse-rotation handled by caller).

    Args:
        key_cache, value_cache: TQ paged cache after kv_cache.unbind(1)
        k_norms, v_norms: Float32 strided views into cache padding
        staging_key, staging_value: bf16 output buffers
        codebook: TurboQuantCodebook with centroids
        block_table: [num_seqs, max_blocks_per_seq] int32
        seq_lens: [num_seqs] int32
        max_blocks_per_seq: Static max blocks per sequence
        k_signs, v_signs: QJL sign bit views (optional)
        k_res_scales, v_res_scales: QJL residual scale views (optional)
        dirty_blocks: [num_blocks] bool — if provided, only decompress
            blocks where dirty_blocks[phys_blk] is True (incremental
            dequant). Pass None for full decompression.
    """
    num_seqs = block_table.shape[0]
    if num_seqs == 0:
        return

    # Ensure centroids are on the right device
    dev = key_cache.device
    centroids = codebook.centroids
    if centroids.device != dev:
        centroids = centroids.to(dev, non_blocking=True)

    qjl = codebook.qjl
    nkv = key_cache.shape[2]
    block_size = key_cache.shape[1]
    dirty_check = dirty_blocks is not None
    grid = (num_seqs * max_blocks_per_seq, nkv)

    items = [
        (key_cache, k_norms, staging_key, k_signs, k_res_scales),
        (value_cache, v_norms, staging_value, v_signs, v_res_scales),
    ]
    for cache, norms, staging, signs, res_scales in items:
        if codebook.byte_mode:
            _turboquant_dequant_byte_kernel[grid](
                cache,
                cache.stride(0),
                cache.stride(1),
                cache.stride(2),
                cache.stride(3),
                norms,
                norms.stride(0),
                norms.stride(1),
                norms.stride(2),
                signs if qjl else cache,
                signs.stride(0) if qjl else 0,
                signs.stride(1) if qjl else 0,
                signs.stride(2) if qjl else 0,
                signs.stride(3) if qjl else 0,
                res_scales if qjl else norms,
                res_scales.stride(0) if qjl else 0,
                res_scales.stride(1) if qjl else 0,
                res_scales.stride(2) if qjl else 0,
                centroids,
                block_table,
                block_table.stride(0),
                block_table.stride(1),
                seq_lens,
                staging,
                staging.stride(0),
                staging.stride(1),
                staging.stride(2),
                staging.stride(3),
                max_blocks_per_seq,
                num_seqs,
                dirty_blocks if dirty_check else seq_lens,
                BLOCK_SIZE=block_size,
                HEAD_DIM=codebook.head_dim,
                QJL_ENABLED=qjl,
                SIGN_BYTES=(codebook.head_dim + 7) // 8 if qjl else 1,
                DIRTY_CHECK=dirty_check,
            )
        else:
            hd = codebook.head_dim
            _turboquant_dequant_nibble_kernel[grid](
                cache,
                cache.stride(0),
                cache.stride(1),
                cache.stride(2),
                cache.stride(3),
                norms,
                norms.stride(0),
                norms.stride(1),
                norms.stride(2),
                signs if qjl else cache,
                signs.stride(0) if qjl else 0,
                signs.stride(1) if qjl else 0,
                signs.stride(2) if qjl else 0,
                signs.stride(3) if qjl else 0,
                res_scales if qjl else norms,
                res_scales.stride(0) if qjl else 0,
                res_scales.stride(1) if qjl else 0,
                res_scales.stride(2) if qjl else 0,
                centroids,
                block_table,
                block_table.stride(0),
                block_table.stride(1),
                seq_lens,
                staging,
                staging.stride(0),
                staging.stride(1),
                staging.stride(2),
                staging.stride(3),
                max_blocks_per_seq,
                num_seqs,
                dirty_blocks if dirty_check else seq_lens,
                BLOCK_SIZE=block_size,
                HEAD_DIM=hd,
                HALF_DIM=hd // 2,
                QJL_ENABLED=qjl,
                SIGN_BYTES_HALF=(hd // 2 + 7) // 8 if qjl else 1,
                DIRTY_CHECK=dirty_check,
            )


# ============================================================================
# Triton Encode Kernel (sign-flip rotation, kept for potential fast path)
# ============================================================================


@triton.jit
# ============================================================================
# Python Reference (for testing)
# ============================================================================


def turboquant_encode_ref(
    tensor: torch.Tensor,
    codebook: TurboQuantCodebook,
    packed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference encode.

    Args:
        packed: If True, return nibble-packed indices (..., d//2).
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    norms = torch.norm(tensor.float(), dim=-1)
    x_hat = tensor.float() / (norms.unsqueeze(-1) + 1e-10)
    R = codebook.rotation_matrix.to(tensor.device)
    y = torch.einsum("...d,de->...e", x_hat, R.T)

    indices = torch.zeros_like(y, dtype=torch.uint8)
    for i in range(codebook.n_levels - 1):
        indices += (y > codebook.boundaries[i].to(tensor.device)).to(torch.uint8)

    # Norm correction: centroid vector is NOT unit norm after coordinate-wise
    # quantization. Correcting ensures ||K_recon|| == original ||K||.
    centroids = codebook.centroids.to(tensor.device)
    centroid_vals = centroids[indices.long()]  # (..., head_dim)
    centroid_norms = torch.norm(centroid_vals.float(), dim=-1)  # (...,)
    corrected_norms = norms / (centroid_norms + 1e-12)

    if packed and not codebook.byte_mode:
        indices = pack_nibbles(indices)

    if squeeze:
        return indices.squeeze(1), corrected_norms.squeeze(1)
    return indices, corrected_norms


def turboquant_decode_ref(
    indices: torch.Tensor,
    norms: torch.Tensor,
    codebook: TurboQuantCodebook,
    output_dtype: torch.dtype = torch.bfloat16,
    packed: bool = True,
) -> torch.Tensor:
    """Reference decode (full inverse rotation).

    Args:
        packed: If True and nibble mode, indices are nibble-packed (..., d//2).
    """
    if indices.dim() == 2:
        indices = indices.unsqueeze(1)
        norms = norms.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    device = indices.device
    centroids = codebook.centroids.to(device)
    R = codebook.rotation_matrix.to(device)

    if packed and not codebook.byte_mode:
        indices = unpack_nibbles(indices, codebook.head_dim)

    y_hat = centroids[indices.long()]
    # Inverse rotation: R^T @ y_hat, in batch = y_hat @ R
    x_hat = torch.einsum("...d,de->...e", y_hat, R)
    x_recon = norms.unsqueeze(-1) * x_hat

    if squeeze:
        return x_recon.squeeze(1).to(output_dtype)
    return x_recon.to(output_dtype)


# ============================================================================
# QJL Reference (for testing)
# ============================================================================


def pack_sign_bits(signs: torch.Tensor) -> torch.Tensor:
    """Pack boolean/0-1 sign tensor into uint8 bytes.

    Args:
        signs: (..., head_dim) bool or uint8 (0/1)
    Returns:
        (..., head_dim // 8) uint8 packed bytes
    """
    d = signs.shape[-1]
    assert d % 8 == 0, f"head_dim must be divisible by 8, got {d}"
    batch_shape = signs.shape[:-1]
    flat = signs.reshape(-1, d).to(torch.uint8)
    n = flat.shape[0]
    reshaped = flat.reshape(n, d // 8, 8)
    bits = torch.arange(8, device=signs.device, dtype=torch.uint8)
    packed = (reshaped << bits[None, None, :]).sum(dim=-1).to(torch.uint8)
    return packed.reshape(*batch_shape, d // 8)


def unpack_sign_bits(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Unpack uint8 bytes to ±1.0 float sign vectors.

    Args:
        packed: (..., head_dim // 8) uint8
    Returns:
        (..., head_dim) float32, values in {-1.0, +1.0}
    """
    batch_shape = packed.shape[:-1]
    n_bytes = packed.shape[-1]
    flat = packed.reshape(-1, n_bytes)
    n = flat.shape[0]
    bits = torch.arange(8, device=packed.device, dtype=torch.uint8)
    unpacked = ((flat.unsqueeze(-1) >> bits[None, None, :]) & 1).reshape(n, n_bytes * 8)
    # Trim to head_dim (may have padding bytes)
    unpacked = unpacked[:, :head_dim]
    sign_vec = 2.0 * unpacked.float() - 1.0
    return sign_vec.reshape(*batch_shape, head_dim)


def turboquant_encode_qjl_ref(
    tensor: torch.Tensor,
    codebook: TurboQuantCodebook,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference QJL encode (practical sign-correction variant).

    Steps:
        1. MSE quantize in rotated space (same as basic TQ)
        2. Compute residual r = y - centroids[idx]
        3. Store sign(r) and mean(|r|)

    Returns:
        indices: (..., head_dim) uint8
        norms: (...) float32 — L2 norms of original vectors
        sign_bits: (..., head_dim // 8) uint8 — packed sign(residual)
        res_scales: (...) float32 — mean(|residual|) per token-head
    """
    assert codebook.qjl, "QJL requires codebook with qjl=True"
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    device = tensor.device
    norms = torch.norm(tensor.float(), dim=-1)
    x_hat = tensor.float() / (norms.unsqueeze(-1) + 1e-10)
    R = codebook.rotation_matrix.to(device)
    y = torch.einsum("...d,de->...e", x_hat, R.T)

    # Scalar quantize
    indices = torch.zeros_like(y, dtype=torch.uint8)
    for i in range(codebook.n_levels - 1):
        indices += (y > codebook.boundaries[i].to(device)).to(torch.uint8)

    # Residual in rotated space
    centroids = codebook.centroids.to(device)
    centroid_vals = centroids[indices.long()]
    residual = y - centroid_vals

    # Norm correction: centroid vector is NOT unit norm after coordinate-wise
    # quantization. Correcting ensures ||K_recon|| == original ||K||.
    centroid_norms = torch.norm(centroid_vals.float(), dim=-1)
    corrected_norms = norms / (centroid_norms + 1e-12)

    # sign(r) and mean(|r|)
    sign_raw = residual >= 0
    sign_bits = pack_sign_bits(sign_raw)
    res_scales = residual.abs().mean(dim=-1)

    if squeeze:
        return (
            indices.squeeze(1),
            corrected_norms.squeeze(1),
            sign_bits.squeeze(1),
            res_scales.squeeze(1),
        )
    return indices, corrected_norms, sign_bits, res_scales


def turboquant_decode_qjl_ref(
    indices: torch.Tensor,
    norms: torch.Tensor,
    sign_bits: torch.Tensor,
    res_scales: torch.Tensor,
    codebook: TurboQuantCodebook,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reference QJL decode (practical sign-correction variant).

    Reconstruction in rotated space:
        y_hat = centroids[idx] + res_scale * sign_vec

    Then inverse-rotate: x_hat = R^T @ y_hat, scale: x = norm * x_hat
    """
    if indices.dim() == 2:
        indices = indices.unsqueeze(1)
        norms = norms.unsqueeze(1)
        sign_bits = sign_bits.unsqueeze(1)
        res_scales = res_scales.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False

    device = indices.device
    centroids = codebook.centroids.to(device)
    R = codebook.rotation_matrix.to(device)

    # MSE reconstruction + sign correction in rotated space
    y_hat = centroids[indices.long()]
    sign_vec = unpack_sign_bits(sign_bits, codebook.head_dim).to(device)
    y_hat = y_hat + res_scales.unsqueeze(-1) * sign_vec

    # Inverse rotation: R^T @ y_hat, in batch = y_hat @ R
    x_hat = torch.einsum("...d,de->...e", y_hat, R)
    x_recon = norms.unsqueeze(-1) * x_hat

    if squeeze:
        return x_recon.squeeze(1).to(output_dtype)
    return x_recon.to(output_dtype)


# ============================================================================
# Single-tensor encode/decode (for standalone backend with separate K/V codebooks)
# ============================================================================


def turboquant_encode_single(
    tensor: torch.Tensor,
    cache: torch.Tensor,
    norms_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    codebook: TurboQuantCodebook,
    signs_cache: torch.Tensor | None = None,
    res_scales_cache: torch.Tensor | None = None,
) -> None:
    """Encode a single tensor (K or V) and scatter into its cache half.

    CUDAGraph-safe: no boolean indexing, no dynamic shapes, no GPU→CPU
    syncs. All tokens are processed; the Triton kernel skips slots < 0.
    """
    num_tokens = tensor.shape[0]
    if num_tokens == 0:
        return

    qjl = codebook.qjl
    R_T = codebook.rotation_matrix_T
    n_heads = tensor.shape[1]

    # Process ALL tokens (no boolean filtering — CUDAGraph requires
    # deterministic tensor shapes). Invalid slots are skipped by the
    # Triton kernel (slot < 0 → early return).
    x = tensor.float()
    nrm = x.norm(dim=-1)
    x_hat = x / (nrm.unsqueeze(-1) + 1e-10)
    y = (x_hat.reshape(-1, x_hat.shape[-1]) @ R_T).reshape(x_hat.shape)

    log2_levels = math.ceil(math.log2(max(codebook.n_levels, 2)))

    grid = (num_tokens, n_heads)
    if codebook.byte_mode:
        _turboquant_encode_byte_kernel[grid](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            cache,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            cache.stride(3),
            norms_cache,
            norms_cache.stride(0),
            norms_cache.stride(1),
            norms_cache.stride(2),
            nrm,
            nrm.stride(0),
            nrm.stride(1),
            slot_mapping,
            codebook.boundaries,
            codebook.centroids,
            signs_cache if qjl else cache,
            signs_cache.stride(0) if qjl else 0,
            signs_cache.stride(1) if qjl else 0,
            signs_cache.stride(2) if qjl else 0,
            signs_cache.stride(3) if qjl else 0,
            res_scales_cache if qjl else norms_cache,
            res_scales_cache.stride(0) if qjl else 0,
            res_scales_cache.stride(1) if qjl else 0,
            res_scales_cache.stride(2) if qjl else 0,
            BLOCK_SIZE=cache.shape[1],
            HEAD_DIM=codebook.head_dim,
            N_LEVELS=codebook.n_levels,
            QJL_ENABLED=qjl,
            SIGN_BYTES=(codebook.head_dim + 7) // 8 if qjl else 1,
            LOG2_LEVELS=log2_levels,
        )
    else:
        hd = codebook.head_dim
        _turboquant_encode_packed_kernel[grid](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            cache,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            cache.stride(3),
            norms_cache,
            norms_cache.stride(0),
            norms_cache.stride(1),
            norms_cache.stride(2),
            nrm,
            nrm.stride(0),
            nrm.stride(1),
            slot_mapping,
            codebook.boundaries,
            codebook.centroids,
            signs_cache if qjl else cache,
            signs_cache.stride(0) if qjl else 0,
            signs_cache.stride(1) if qjl else 0,
            signs_cache.stride(2) if qjl else 0,
            signs_cache.stride(3) if qjl else 0,
            res_scales_cache if qjl else norms_cache,
            res_scales_cache.stride(0) if qjl else 0,
            res_scales_cache.stride(1) if qjl else 0,
            res_scales_cache.stride(2) if qjl else 0,
                BLOCK_SIZE=cache.shape[1],
                HEAD_DIM=hd,
                N_LEVELS=codebook.n_levels,
                HALF_DIM=hd // 2,
                QJL_ENABLED=qjl,
                SIGN_BYTES_HALF=(hd // 2 + 7) // 8 if qjl else 1,
                LOG2_LEVELS=log2_levels,
            )


def turboquant_dequant_single(
    cache: torch.Tensor,
    norms: torch.Tensor,
    staging: torch.Tensor,
    codebook: TurboQuantCodebook,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_blocks_per_seq: int,
    signs: torch.Tensor | None = None,
    res_scales: torch.Tensor | None = None,
    dirty_blocks: torch.Tensor | None = None,
) -> None:
    """Decompress a single cache half (K or V) to bf16 staging buffer."""
    num_seqs = block_table.shape[0]
    if num_seqs == 0:
        return

    dev = cache.device
    centroids = codebook.centroids
    if centroids.device != dev:
        centroids = centroids.to(dev, non_blocking=True)

    qjl = codebook.qjl
    nkv = cache.shape[2]
    block_size = cache.shape[1]
    dirty_check = dirty_blocks is not None
    grid = (num_seqs * max_blocks_per_seq, nkv)

    if codebook.byte_mode:
        _turboquant_dequant_byte_kernel[grid](
            cache,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            cache.stride(3),
            norms,
            norms.stride(0),
            norms.stride(1),
            norms.stride(2),
            signs if qjl else cache,
            signs.stride(0) if qjl else 0,
            signs.stride(1) if qjl else 0,
            signs.stride(2) if qjl else 0,
            signs.stride(3) if qjl else 0,
            res_scales if qjl else norms,
            res_scales.stride(0) if qjl else 0,
            res_scales.stride(1) if qjl else 0,
            res_scales.stride(2) if qjl else 0,
            centroids,
            block_table,
            block_table.stride(0),
            block_table.stride(1),
            seq_lens,
            staging,
            staging.stride(0),
            staging.stride(1),
            staging.stride(2),
            staging.stride(3),
            max_blocks_per_seq,
            num_seqs,
            dirty_blocks if dirty_check else seq_lens,
            BLOCK_SIZE=block_size,
            HEAD_DIM=codebook.head_dim,
            QJL_ENABLED=qjl,
            SIGN_BYTES=(codebook.head_dim + 7) // 8 if qjl else 1,
            DIRTY_CHECK=dirty_check,
        )
    else:
        hd = codebook.head_dim
        _turboquant_dequant_nibble_kernel[grid](
            cache,
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            cache.stride(3),
            norms,
            norms.stride(0),
            norms.stride(1),
            norms.stride(2),
            signs if qjl else cache,
            signs.stride(0) if qjl else 0,
            signs.stride(1) if qjl else 0,
            signs.stride(2) if qjl else 0,
            signs.stride(3) if qjl else 0,
            res_scales if qjl else norms,
            res_scales.stride(0) if qjl else 0,
            res_scales.stride(1) if qjl else 0,
            res_scales.stride(2) if qjl else 0,
            centroids,
            block_table,
            block_table.stride(0),
            block_table.stride(1),
            seq_lens,
            staging,
            staging.stride(0),
            staging.stride(1),
            staging.stride(2),
            staging.stride(3),
            max_blocks_per_seq,
            num_seqs,
            dirty_blocks if dirty_check else seq_lens,
            BLOCK_SIZE=block_size,
            HEAD_DIM=hd,
            HALF_DIM=hd // 2,
            QJL_ENABLED=qjl,
            SIGN_BYTES_HALF=(hd // 2 + 7) // 8 if qjl else 1,
            DIRTY_CHECK=dirty_check,
        )
