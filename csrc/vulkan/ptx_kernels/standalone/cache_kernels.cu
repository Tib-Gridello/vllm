// SPDX-License-Identifier: Apache-2.0
// Standalone vLLM KV cache kernels for PTX extraction.
//
// Extracted from vllm/csrc/cache_kernels.cu, stripped of ATen/FP8 deps.
// Covers the core cache operations needed for inference.
//
// Compile: nvcc -ptx -arch=sm_80 -O3 cache_kernels.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================================
// copy_blocks: Copy KV cache blocks between locations
// Grid: (num_layers, num_pairs)
// ============================================================================

template <typename scalar_t>
__device__ __forceinline__ void copy_blocks_kernel_impl(
    int64_t* key_cache_ptrs,                       // [num_layers] pointers
    int64_t* value_cache_ptrs,                     // [num_layers] pointers
    const int64_t* __restrict__ block_mapping,     // [num_pairs, 2]
    const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block = block_mapping[2 * pair_idx];
  int64_t dst_block = block_mapping[2 * pair_idx + 1];

  const int64_t src_offset = src_block * numel_per_block;
  const int64_t dst_offset = dst_block * numel_per_block;

  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    key_cache[dst_offset + i] = key_cache[src_offset + i];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    value_cache[dst_offset + i] = value_cache[src_offset + i];
  }
}

extern "C" __global__
void copy_blocks_f16(int64_t* key_cache_ptrs, int64_t* value_cache_ptrs,
                     const int64_t* block_mapping, int numel_per_block) {
  copy_blocks_kernel_impl<__half>(key_cache_ptrs, value_cache_ptrs,
                                  block_mapping, numel_per_block);
}

extern "C" __global__
void copy_blocks_bf16(int64_t* key_cache_ptrs, int64_t* value_cache_ptrs,
                      const int64_t* block_mapping, int numel_per_block) {
  copy_blocks_kernel_impl<__nv_bfloat16>(key_cache_ptrs, value_cache_ptrs,
                                         block_mapping, numel_per_block);
}

// ============================================================================
// reshape_and_cache: Write new KV pairs into the paged cache
//
// Key layout:   [num_blocks, num_heads, head_size/x, block_size, x]
// Value layout: [num_blocks, num_heads, head_size, block_size]
// where x = 16 / sizeof(scalar_t) (packing factor for coalesced access)
// ============================================================================

template <typename scalar_t>
__device__ __forceinline__ void reshape_and_cache_kernel_impl(
    const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
    scalar_t* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const int key_stride,                       // num_heads * head_size
    const int value_stride,                     // num_heads * head_size
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x) {                              // packing factor = 16 / sizeof(scalar_t)
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  // Padding tokens have slot_idx < 0
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    // Key: [num_blocks, num_heads, head_size/x, block_size, x]
    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +
        head_idx * (head_size / x) * block_size * x +
        x_idx * block_size * x +
        block_offset * x +
        x_offset;

    // Value: [num_blocks, num_heads, head_size, block_size]
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +
        head_idx * head_size * block_size +
        head_offset * block_size +
        block_offset;

    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}

extern "C" __global__
void reshape_and_cache_f16(
    const __half* key, const __half* value,
    __half* key_cache, __half* value_cache,
    const int64_t* slot_mapping,
    int key_stride, int value_stride,
    int num_heads, int head_size, int block_size, int x) {
  reshape_and_cache_kernel_impl<__half>(
      key, value, key_cache, value_cache, slot_mapping,
      key_stride, value_stride, num_heads, head_size, block_size, x);
}

extern "C" __global__
void reshape_and_cache_bf16(
    const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
    const int64_t* slot_mapping,
    int key_stride, int value_stride,
    int num_heads, int head_size, int block_size, int x) {
  reshape_and_cache_kernel_impl<__nv_bfloat16>(
      key, value, key_cache, value_cache, slot_mapping,
      key_stride, value_stride, num_heads, head_size, block_size, x);
}
