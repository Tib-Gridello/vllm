// SPDX-License-Identifier: Apache-2.0
// Standalone vLLM PagedAttention v1 kernel for PTX extraction.
//
// Adapted from vllm/csrc/attention/attention_kernels.cuh
// Stripped of ATen dependencies, supports f16/bf16 with non-quantized KV cache.
//
// This is the decode-path attention kernel (one query token per sequence).
// For prefill, FlashAttention is typically used instead.
//
// Compile: nvcc -ptx -arch=sm_80 -O3 paged_attention_kernels.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

#define WARP_SIZE 32
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

// ============================================================================
// Warp-level primitives
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int mask = WARP_SIZE / 2; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int mask = WARP_SIZE / 2; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block-level sum reduction
template <int NUM_WARPS>
__device__ float block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  sum = warp_reduce_sum(sum);
  if (lane == 0) red_smem[warp] = sum;
  __syncthreads();

  if (lane < NUM_WARPS) sum = red_smem[lane];
  else sum = 0.0f;

  for (int mask = NUM_WARPS / 2; mask >= 1; mask >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, mask);
  }
  return __shfl_sync(0xffffffff, sum, 0);
}

// Block-level max reduction
template <int NUM_WARPS>
__device__ float block_max(float* red_smem, float val) {
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  val = warp_reduce_max(val);
  if (lane == 0) red_smem[warp] = val;
  __syncthreads();

  if (lane < NUM_WARPS) val = red_smem[lane];
  else val = -FLT_MAX;

  for (int mask = NUM_WARPS / 2; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return __shfl_sync(0xffffffff, val, 0);
}

// ============================================================================
// PagedAttention V1 kernel (non-partitioned decode)
//
// Each thread block handles one head of one sequence.
// Grid: (num_heads, num_seqs)
//
// Parameters:
//   out:          [num_seqs, num_heads, head_size]
//   q:            [num_seqs, num_heads, head_size]
//   k_cache:      [num_blocks, num_kv_heads, head_size/x, block_size, x]
//   v_cache:      [num_blocks, num_kv_heads, head_size, block_size]
//   block_tables:  [num_seqs, max_num_blocks_per_seq]
//   seq_lens:      [num_seqs]
//
// Template params:
//   HEAD_SIZE:    dimension of each attention head (e.g. 128)
//   BLOCK_SIZE:   number of KV entries per cache block (e.g. 16)
//   NUM_THREADS:  threads per block (e.g. 128)
// ============================================================================

template <int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__device__ __forceinline__ void paged_attention_v1_kernel(
    float* __restrict__ out,                   // [num_seqs, num_heads, head_size]
    const float* __restrict__ q,               // [num_seqs, num_heads, head_size]
    const float* __restrict__ k_cache,         // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const float* __restrict__ v_cache,         // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,      // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,          // [num_seqs]
    const int max_num_blocks_per_seq,
    const int q_stride,                        // num_heads * head_size
    const int kv_block_stride,                 // num_kv_heads * head_size * block_size
    const int kv_head_stride) {                // head_size * block_size

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int x = 16 / sizeof(float);  // packing factor (4 for f32, 8 for f16)

  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  if (seq_len == 0) return;

  const int num_blocks_for_seq = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int kv_head_idx = head_idx % num_kv_heads;

  // Load query vector into registers
  float q_vec[HEAD_SIZE / NUM_THREADS + 1];
  const float* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  #pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    q_vec[i / NUM_THREADS] = q_ptr[i] * scale;
  }

  // Shared memory for softmax reduction
  __shared__ float red_smem[NUM_WARPS];

  // Compute attention scores and accumulate weighted values
  // We process one KV cache block at a time
  float acc[HEAD_SIZE / NUM_THREADS + 1] = {0.0f};
  float global_max = -FLT_MAX;
  float global_sum = 0.0f;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_i = 0; block_i < num_blocks_for_seq; block_i++) {
    const int physical_block = block_table[block_i];
    const int tokens_in_block = (block_i == num_blocks_for_seq - 1)
                                    ? seq_len - block_i * BLOCK_SIZE
                                    : BLOCK_SIZE;

    // Compute QK^T for this block
    // Each thread computes dot products for a subset of positions
    for (int pos_in_block = 0; pos_in_block < tokens_in_block; pos_in_block++) {
      float qk = 0.0f;
      const float* k_ptr = k_cache +
          physical_block * kv_block_stride +
          kv_head_idx * kv_head_stride;

      // Simplified dot product (without the x-packing optimization)
      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        // For f32 with x=4: key layout is [head_size/4, block_size, 4]
        int x_idx = d / x;
        int x_offset = d % x;
        int k_idx = x_idx * BLOCK_SIZE * x + pos_in_block * x + x_offset;
        qk += q_vec[d / NUM_THREADS] * k_ptr[k_idx];
      }

      // Reduce QK across threads
      qk = block_sum<NUM_WARPS>(red_smem, qk);

      // Online softmax update
      float old_max = global_max;
      global_max = fmaxf(global_max, qk);
      float exp_correction = expf(old_max - global_max);
      global_sum = global_sum * exp_correction + expf(qk - global_max);

      // Correct accumulated values for the new max
      #pragma unroll
      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        acc[d / NUM_THREADS] *= exp_correction;
      }

      // Accumulate V weighted by attention score
      float weight = expf(qk - global_max);
      const float* v_ptr = v_cache +
          physical_block * kv_block_stride +
          kv_head_idx * kv_head_stride;

      #pragma unroll
      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        // Value layout: [head_size, block_size]
        int v_idx = d * BLOCK_SIZE + pos_in_block;
        acc[d / NUM_THREADS] += weight * v_ptr[v_idx];
      }
    }
  }

  // Normalize by sum of exp(scores)
  float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

  // Write output
  float* out_ptr = out + seq_idx * q_stride + head_idx * HEAD_SIZE;
  #pragma unroll
  for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
    out_ptr[d] = acc[d / NUM_THREADS] * inv_sum;
  }
}

// ============================================================================
// Explicit instantiations for common configurations
// ============================================================================

// HEAD_SIZE=128, BLOCK_SIZE=16, NUM_THREADS=128 (LLaMA, Mistral, etc.)
extern "C" __global__
void paged_attention_v1_h128_b16_t128(
    float* out, const float* q,
    const float* k_cache, const float* v_cache,
    int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens,
    int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride) {
  paged_attention_v1_kernel<128, 16, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride);
}

// HEAD_SIZE=64, BLOCK_SIZE=16, NUM_THREADS=128 (smaller models)
extern "C" __global__
void paged_attention_v1_h64_b16_t128(
    float* out, const float* q,
    const float* k_cache, const float* v_cache,
    int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens,
    int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride) {
  paged_attention_v1_kernel<64, 16, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride);
}

// HEAD_SIZE=128, BLOCK_SIZE=32, NUM_THREADS=128
extern "C" __global__
void paged_attention_v1_h128_b32_t128(
    float* out, const float* q,
    const float* k_cache, const float* v_cache,
    int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens,
    int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride) {
  paged_attention_v1_kernel<128, 32, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride);
}

// ============================================================================
// Half-precision (f16) PagedAttention V1
//
// Input/output in __half, KV cache in __half, accumulation in float.
// This is the primary dtype for LLM inference.
// ============================================================================

template <int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__device__ __forceinline__ void paged_attention_v1_f16_kernel(
    __half* __restrict__ out,
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int x = 16 / sizeof(__half);  // 8 for f16

  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  if (seq_len == 0) return;

  const int num_blocks_for_seq = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int kv_head_idx = head_idx % num_kv_heads;

  // Load query into float registers
  float q_vec[HEAD_SIZE / NUM_THREADS + 1];
  const __half* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    q_vec[i / NUM_THREADS] = __half2float(q_ptr[i]) * scale;
  }

  __shared__ float red_smem[NUM_WARPS];

  float acc[HEAD_SIZE / NUM_THREADS + 1] = {0.0f};
  float global_max = -FLT_MAX;
  float global_sum = 0.0f;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_i = 0; block_i < num_blocks_for_seq; block_i++) {
    const int physical_block = block_table[block_i];
    const int tokens_in_block = (block_i == num_blocks_for_seq - 1)
                                    ? seq_len - block_i * BLOCK_SIZE
                                    : BLOCK_SIZE;

    for (int pos_in_block = 0; pos_in_block < tokens_in_block; pos_in_block++) {
      float qk = 0.0f;
      const __half* k_ptr = k_cache +
          physical_block * kv_block_stride +
          kv_head_idx * kv_head_stride;

      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        int x_idx = d / x;
        int x_offset = d % x;
        int k_idx = x_idx * BLOCK_SIZE * x + pos_in_block * x + x_offset;
        qk += q_vec[d / NUM_THREADS] * __half2float(k_ptr[k_idx]);
      }

      qk = block_sum<NUM_WARPS>(red_smem, qk);

      float old_max = global_max;
      global_max = fmaxf(global_max, qk);
      float exp_correction = expf(old_max - global_max);
      global_sum = global_sum * exp_correction + expf(qk - global_max);

      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        acc[d / NUM_THREADS] *= exp_correction;
      }

      float weight = expf(qk - global_max);
      const __half* v_ptr = v_cache +
          physical_block * kv_block_stride +
          kv_head_idx * kv_head_stride;

      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        int v_idx = d * BLOCK_SIZE + pos_in_block;
        acc[d / NUM_THREADS] += weight * __half2float(v_ptr[v_idx]);
      }
    }
  }

  float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

  __half* out_ptr = out + seq_idx * q_stride + head_idx * HEAD_SIZE;
  for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
    out_ptr[d] = __float2half(acc[d / NUM_THREADS] * inv_sum);
  }
}

// f16 instantiations
extern "C" __global__
void paged_attention_v1_f16_h128_b16_t128(
    __half* out, const __half* q,
    const __half* k_cache, const __half* v_cache,
    int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens,
    int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride) {
  paged_attention_v1_f16_kernel<128, 16, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride);
}

extern "C" __global__
void paged_attention_v1_f16_h64_b16_t128(
    __half* out, const __half* q,
    const __half* k_cache, const __half* v_cache,
    int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens,
    int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride) {
  paged_attention_v1_f16_kernel<64, 16, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride);
}

// ============================================================================
// BFloat16 PagedAttention V1
// ============================================================================

template <int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__device__ __forceinline__ void paged_attention_v1_bf16_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int x = 16 / sizeof(__nv_bfloat16);  // 8 for bf16

  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  if (seq_len == 0) return;

  const int num_blocks_for_seq = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int kv_head_idx = head_idx % num_kv_heads;

  float q_vec[HEAD_SIZE / NUM_THREADS + 1];
  const __nv_bfloat16* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    q_vec[i / NUM_THREADS] = __bfloat162float(q_ptr[i]) * scale;
  }

  __shared__ float red_smem[NUM_WARPS];

  float acc[HEAD_SIZE / NUM_THREADS + 1] = {0.0f};
  float global_max = -FLT_MAX;
  float global_sum = 0.0f;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  for (int block_i = 0; block_i < num_blocks_for_seq; block_i++) {
    const int physical_block = block_table[block_i];
    const int tokens_in_block = (block_i == num_blocks_for_seq - 1)
                                    ? seq_len - block_i * BLOCK_SIZE
                                    : BLOCK_SIZE;

    for (int pos_in_block = 0; pos_in_block < tokens_in_block; pos_in_block++) {
      float qk = 0.0f;
      const __nv_bfloat16* k_ptr = k_cache +
          physical_block * kv_block_stride +
          kv_head_idx * kv_head_stride;

      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        int x_idx = d / x;
        int x_offset = d % x;
        int k_idx = x_idx * BLOCK_SIZE * x + pos_in_block * x + x_offset;
        qk += q_vec[d / NUM_THREADS] * __bfloat162float(k_ptr[k_idx]);
      }

      qk = block_sum<NUM_WARPS>(red_smem, qk);

      float old_max = global_max;
      global_max = fmaxf(global_max, qk);
      float exp_correction = expf(old_max - global_max);
      global_sum = global_sum * exp_correction + expf(qk - global_max);

      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        acc[d / NUM_THREADS] *= exp_correction;
      }

      float weight = expf(qk - global_max);
      const __nv_bfloat16* v_ptr = v_cache +
          physical_block * kv_block_stride +
          kv_head_idx * kv_head_stride;

      for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
        int v_idx = d * BLOCK_SIZE + pos_in_block;
        acc[d / NUM_THREADS] += weight * __bfloat162float(v_ptr[v_idx]);
      }
    }
  }

  float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

  __nv_bfloat16* out_ptr = out + seq_idx * q_stride + head_idx * HEAD_SIZE;
  for (int d = threadIdx.x; d < HEAD_SIZE; d += NUM_THREADS) {
    out_ptr[d] = __float2bfloat16(acc[d / NUM_THREADS] * inv_sum);
  }
}

// bf16 instantiations
extern "C" __global__
void paged_attention_v1_bf16_h128_b16_t128(
    __nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens,
    int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride) {
  paged_attention_v1_bf16_kernel<128, 16, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride);
}

extern "C" __global__
void paged_attention_v1_bf16_h64_b16_t128(
    __nv_bfloat16* out, const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int num_kv_heads, float scale,
    const int* block_tables, const int* seq_lens,
    int max_num_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride) {
  paged_attention_v1_bf16_kernel<64, 16, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq,
      q_stride, kv_block_stride, kv_head_stride);
}
