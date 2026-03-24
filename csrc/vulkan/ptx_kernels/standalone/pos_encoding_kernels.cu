// SPDX-License-Identifier: Apache-2.0
// Standalone vLLM Rotary Position Encoding (RoPE) kernels for PTX extraction.
//
// Extracted from vllm/csrc/pos_encoding_kernels.cu, stripped of ATen deps.
// Supports both GPT-NeoX and GPT-J style rotary embeddings.
//
// Compile: nvcc -ptx -arch=sm_80 -O3 pos_encoding_kernels.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// Device helper: apply rotary embedding to a single (x, y) pair
// ============================================================================

template <typename scalar_t, bool IS_NEOX>
__device__ __forceinline__ void apply_token_rotary(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr,
    int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t cos_val, sin_val;
  if constexpr (IS_NEOX) {
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos_val = cos_ptr[x_index];
    sin_val = sin_ptr[x_index];
  } else {
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos_val = cos_ptr[x_index / 2];
    sin_val = sin_ptr[x_index / 2];
  }
  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos_val - y * sin_val;
  arr[y_index] = y * cos_val + x * sin_val;
}

// ============================================================================
// RoPE kernel: applies rotary embedding to query and optionally key
// ============================================================================

template <typename scalar_t, bool IS_NEOX>
__device__ __forceinline__ void rotary_embedding_kernel_impl(
    const int64_t* __restrict__ positions,       // [num_tokens]
    scalar_t* __restrict__ query,                // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key,                  // [num_tokens, num_kv_heads, head_size] or nullptr
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t head_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cos_sin_cache + pos * rot_dim;
  const scalar_t* sin_ptr = cos_ptr + embed_dim;

  // Apply to query heads
  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_stride;
    const int rot_offset = i % embed_dim;
    apply_token_rotary<scalar_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  // Apply to key heads (if key is not null)
  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head = token_idx * key_stride + head_idx * head_stride;
      const int rot_offset = i % embed_dim;
      apply_token_rotary<scalar_t, IS_NEOX>(
          key + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
    }
  }
}

// ============================================================================
// Explicit instantiations with extern "C" for PTX extraction
// ============================================================================

// NeoX-style (used by LLaMA, Mistral, etc.)
extern "C" __global__
void rotary_embedding_neox_f32(
    const int64_t* positions, float* query, float* key,
    const float* cos_sin_cache,
    int rot_dim, int64_t query_stride, int64_t key_stride,
    int64_t head_stride, int num_heads, int num_kv_heads, int head_size) {
  rotary_embedding_kernel_impl<float, true>(
      positions, query, key, cos_sin_cache,
      rot_dim, query_stride, key_stride, head_stride,
      num_heads, num_kv_heads, head_size);
}

extern "C" __global__
void rotary_embedding_neox_f16(
    const int64_t* positions, __half* query, __half* key,
    const __half* cos_sin_cache,
    int rot_dim, int64_t query_stride, int64_t key_stride,
    int64_t head_stride, int num_heads, int num_kv_heads, int head_size) {
  rotary_embedding_kernel_impl<__half, true>(
      positions, query, key, cos_sin_cache,
      rot_dim, query_stride, key_stride, head_stride,
      num_heads, num_kv_heads, head_size);
}

extern "C" __global__
void rotary_embedding_neox_bf16(
    const int64_t* positions, __nv_bfloat16* query, __nv_bfloat16* key,
    const __nv_bfloat16* cos_sin_cache,
    int rot_dim, int64_t query_stride, int64_t key_stride,
    int64_t head_stride, int num_heads, int num_kv_heads, int head_size) {
  rotary_embedding_kernel_impl<__nv_bfloat16, true>(
      positions, query, key, cos_sin_cache,
      rot_dim, query_stride, key_stride, head_stride,
      num_heads, num_kv_heads, head_size);
}

// GPT-J style
extern "C" __global__
void rotary_embedding_gptj_f16(
    const int64_t* positions, __half* query, __half* key,
    const __half* cos_sin_cache,
    int rot_dim, int64_t query_stride, int64_t key_stride,
    int64_t head_stride, int num_heads, int num_kv_heads, int head_size) {
  rotary_embedding_kernel_impl<__half, false>(
      positions, query, key, cos_sin_cache,
      rot_dim, query_stride, key_stride, head_stride,
      num_heads, num_kv_heads, head_size);
}

extern "C" __global__
void rotary_embedding_gptj_bf16(
    const int64_t* positions, __nv_bfloat16* query, __nv_bfloat16* key,
    const __nv_bfloat16* cos_sin_cache,
    int rot_dim, int64_t query_stride, int64_t key_stride,
    int64_t head_stride, int num_heads, int num_kv_heads, int head_size) {
  rotary_embedding_kernel_impl<__nv_bfloat16, false>(
      positions, query, key, cos_sin_cache,
      rot_dim, query_stride, key_stride, head_stride,
      num_heads, num_kv_heads, head_size);
}
