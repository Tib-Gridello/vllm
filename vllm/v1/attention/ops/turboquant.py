# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TurboQuant KV Cache Compression

Implements PolarQuant from "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate" (Google Research, ICLR 2026, arXiv:2504.19874).

Algorithm (matching the paper exactly):
  1. Random orthogonal rotation R via QR decomposition
  2. Normalize K/V to unit vectors, store L2 norm separately
  3. Rotate: y = R @ x_hat (coordinates become ~independent Beta(d/2))
  4. Lloyd-Max scalar quantize each coordinate for Beta(d/2) on [-1,1]
  5. At attention time: rotate Q by R (Q_rot = Q @ R^T), no inverse
     rotation needed for K. Inverse-rotate output by R^T for V.

Key identity: q · k_recon = ||k|| · (R @ q)^T · centroids[idx_k]
"""

import math

import torch

from vllm.triton_utils import tl, triton


# ============================================================================
# Codebook Computation (CPU, one-time)
# ============================================================================


def compute_beta_centroids(
    d: int,
    n_bits: int,
    n_iters: int = 200,
    n_samples: int = 200000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd-Max centroids for Beta(d/2) on [-1, 1].

    This is the CORRECT distribution for coordinates of a randomly rotated
    unit vector in R^d. The paper uses this, NOT N(0,1).

    Returns:
        boundaries: (2^n_bits - 1,) decision boundaries
        centroids: (2^n_bits,) reconstruction centroids
    """
    n_levels = 2**n_bits
    alpha = (d - 1) / 2.0

    # Sample from Beta distribution mapped to [-1, 1]
    beta_dist = torch.distributions.Beta(alpha, alpha)
    samples = beta_dist.sample((n_samples,)).double() * 2 - 1

    # Initialize centroids
    centroids = torch.linspace(-0.95, 0.95, n_levels, dtype=torch.float64)

    for _ in range(n_iters):
        dists = (samples.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)

        new_centroids = torch.zeros_like(centroids)
        for i in range(n_levels):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = samples[mask].mean()
            else:
                new_centroids[i] = centroids[i]
        centroids = new_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return boundaries.float(), centroids.float()


def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """Random orthogonal matrix via QR decomposition (paper's method).

    Returns:
        R: (d, d) orthogonal matrix where R @ R^T = I
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    M = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)
    Q, R_qr = torch.linalg.qr(M)
    Q = Q * torch.diag(R_qr).sign().unsqueeze(0)
    return Q


# ============================================================================
# TurboQuant Codebook
# ============================================================================


class TurboQuantCodebook:
    """Pre-computed TurboQuant codebook matching the paper exactly."""

    def __init__(
        self,
        n_bits: int = 4,
        head_dim: int = 128,
        seed: int = 42,
        device: str = "cuda",
    ):
        if n_bits > 4:
            raise ValueError(
                f"TurboQuant nibble packing requires n_bits <= 4, got {n_bits}. "
                f"Set VLLM_TURBOQUANT_BITS=4 or VLLM_TURBOQUANT_BITS=3."
            )
        self.n_bits = n_bits
        self.n_levels = 2**n_bits
        self.head_dim = head_dim
        self.seed = seed

        # Beta(d/2) centroids on [-1, 1] (NOT N(0,1)!)
        boundaries, centroids = compute_beta_centroids(head_dim, n_bits)
        self.boundaries = boundaries.contiguous().to(device)
        self.centroids = centroids.contiguous().to(device)

        # Full orthogonal rotation (NOT sign flips!)
        R = generate_rotation_matrix(head_dim, seed)
        self.rotation_matrix = R.contiguous().to(device)

    def to(self, device) -> "TurboQuantCodebook":
        self.boundaries = self.boundaries.to(device)
        self.centroids = self.centroids.to(device)
        self.rotation_matrix = self.rotation_matrix.to(device)
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
    lo = indices[..., :d // 2]
    hi = indices[..., d // 2:]
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
    # Quantization boundaries
    boundaries_ptr,
    # Constants
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    N_LEVELS: tl.constexpr,
    HALF_DIM: tl.constexpr,
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
    norm = tl.load(orig_norms_ptr + tok * stride_on_tok
                   + head * stride_on_head)

    # Load both halves of rotated normalized vector
    base = tok * stride_r_tok + head * stride_r_head
    offs_lo = tl.arange(0, HALF_DIM)
    offs_hi = HALF_DIM + tl.arange(0, HALF_DIM)

    y_lo = tl.load(rotated_ptr + base + offs_lo * stride_r_dim).to(
        tl.float32)
    y_hi = tl.load(rotated_ptr + base + offs_hi * stride_r_dim).to(
        tl.float32)

    # Scalar quantize each half via boundary scan
    idx_lo = tl.zeros([HALF_DIM], dtype=tl.uint8)
    idx_hi = tl.zeros([HALF_DIM], dtype=tl.uint8)
    for i in range(N_LEVELS - 1):
        b = tl.load(boundaries_ptr + i)
        idx_lo += (y_lo > b).to(tl.uint8)
        idx_hi += (y_hi > b).to(tl.uint8)

    # Pack nibbles: low half → low nibble, high half → high nibble
    packed = idx_lo | (idx_hi << 4)

    # Scatter packed indices to cache
    c_base = (blk * stride_c_blk + off * stride_c_slot
              + head * stride_c_head)
    tl.store(cache_ptr + c_base + offs_lo * stride_c_dim, packed)

    # Scatter norm (float32 via strided view)
    n_off = blk * stride_n_blk + off * stride_n_slot + head * stride_n_head
    tl.store(norms_ptr + n_off, norm)


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

    R = codebook.rotation_matrix  # (d, d)

    valid = slot_mapping >= 0
    valid_slots = slot_mapping[valid]

    for src, cache, norms_cache in [(key, key_cache, k_norms),
                                     (value, value_cache, v_norms)]:
        x = src[valid].float()  # (n_valid, num_heads, head_dim)

        # 1. Compute norms (PyTorch)
        nrm = x.norm(dim=-1)  # (n_valid, num_heads)

        # 2. Normalize to unit vectors (PyTorch)
        x_hat = x / (nrm.unsqueeze(-1) + 1e-10)

        # 3. Rotate: y = R @ x_hat (cuBLAS matmul)
        y = torch.einsum('...d,de->...e', x_hat, R.T)

        # 4+5. Quantize + pack + scatter (Triton kernel)
        n_valid = x.shape[0]
        n_heads = x.shape[1]
        if n_valid > 0:
            grid = (n_valid, n_heads)
            _turboquant_encode_packed_kernel[grid](
                y, y.stride(0), y.stride(1), y.stride(2),
                cache, cache.stride(0), cache.stride(1),
                cache.stride(2), cache.stride(3),
                norms_cache, norms_cache.stride(0),
                norms_cache.stride(1), norms_cache.stride(2),
                nrm, nrm.stride(0), nrm.stride(1),
                valid_slots, codebook.boundaries,
                BLOCK_SIZE=cache.shape[1],
                HEAD_DIM=codebook.head_dim,
                N_LEVELS=codebook.n_levels,
                HALF_DIM=codebook.head_dim // 2,
            )


def rotate_query(query: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate query by R: q_rot = R @ q, in batch = q @ R^T.

    This avoids inverse-rotating K in the kernel.
    Identity: q · k_recon = ||k|| · (R @ q)^T · centroids[idx_k]
    """
    return torch.einsum('...d,de->...e', query.float(), R.T).to(query.dtype)


def inverse_rotate_output(output: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Inverse-rotate attention output: R^T @ out, in batch = out @ R.

    Needed because V is in rotated space.
    """
    return torch.einsum('...d,de->...e', output.float(), R).to(output.dtype)


# ============================================================================
# Triton Encode Kernel (sign-flip rotation, kept for potential fast path)
# ============================================================================


@triton.jit
def _turboquant_encode_signflip_kernel(
    key_ptr, value_ptr,
    input_stride_token: tl.int64, input_stride_head: tl.int64,
    key_cache_ptr, value_cache_ptr,
    cache_stride_block: tl.int64, cache_stride_slot: tl.int64,
    cache_stride_head: tl.int64,
    k_norms_ptr, v_norms_ptr,
    norms_stride_block: tl.int64, norms_stride_slot: tl.int64,
    norms_stride_head: tl.int64,
    slot_mapping_ptr,
    boundaries_ptr, rotation_signs_ptr,
    N_LEVELS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Triton encode kernel with sign-flip rotation (fast but less accurate)."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot < 0:
        return
    block_idx = slot // BLOCK_SIZE
    block_offset = slot % BLOCK_SIZE
    offs = tl.arange(0, HEAD_DIM)
    src_offset = token_idx * input_stride_token + head_idx * input_stride_head
    cache_offset = (block_idx * cache_stride_block
                    + block_offset * cache_stride_slot
                    + head_idx * cache_stride_head)
    norms_offset = (block_idx * norms_stride_block
                    + block_offset * norms_stride_slot
                    + head_idx * norms_stride_head)
    signs = tl.load(rotation_signs_ptr + offs)
    k = tl.load(key_ptr + src_offset + offs).to(tl.float32)
    k_norm = tl.sqrt(tl.sum(k * k, axis=0) + 1e-12)
    k_rot = k / (k_norm + 1e-8) * signs
    k_idx = tl.zeros([HEAD_DIM], dtype=tl.uint8)
    for i in range(N_LEVELS - 1):
        b = tl.load(boundaries_ptr + i)
        k_idx += (k_rot > b).to(tl.uint8)
    tl.store(key_cache_ptr + cache_offset + offs, k_idx)
    tl.store(k_norms_ptr + norms_offset, k_norm)
    v = tl.load(value_ptr + src_offset + offs).to(tl.float32)
    v_norm = tl.sqrt(tl.sum(v * v, axis=0) + 1e-12)
    v_rot = v / (v_norm + 1e-8) * signs
    v_idx = tl.zeros([HEAD_DIM], dtype=tl.uint8)
    for i in range(N_LEVELS - 1):
        b = tl.load(boundaries_ptr + i)
        v_idx += (v_rot > b).to(tl.uint8)
    tl.store(value_cache_ptr + cache_offset + offs, v_idx)
    tl.store(v_norms_ptr + norms_offset, v_norm)


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
    y = torch.einsum('...d,de->...e', x_hat, R.T)

    indices = torch.zeros_like(y, dtype=torch.uint8)
    for i in range(codebook.n_levels - 1):
        indices += (y > codebook.boundaries[i].to(tensor.device)).to(
            torch.uint8)

    if packed:
        indices = pack_nibbles(indices)

    if squeeze:
        return indices.squeeze(1), norms.squeeze(1)
    return indices, norms


def turboquant_decode_ref(
    indices: torch.Tensor,
    norms: torch.Tensor,
    codebook: TurboQuantCodebook,
    output_dtype: torch.dtype = torch.bfloat16,
    packed: bool = True,
) -> torch.Tensor:
    """Reference decode (full inverse rotation).

    Args:
        packed: If True, indices are nibble-packed (..., d//2).
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

    if packed:
        indices = unpack_nibbles(indices, codebook.head_dim)

    y_hat = centroids[indices.long()]
    # Inverse rotation: R^T @ y_hat, in batch = y_hat @ R
    x_hat = torch.einsum('...d,de->...e', y_hat, R)
    x_recon = norms.unsqueeze(-1) * x_hat

    if squeeze:
        return x_recon.squeeze(1).to(output_dtype)
    return x_recon.to(output_dtype)


# Keep old name for backward compat
generate_rotation_signs = None  # removed, use generate_rotation_matrix
