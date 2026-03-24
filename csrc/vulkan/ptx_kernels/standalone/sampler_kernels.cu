// SPDX-License-Identifier: Apache-2.0
// Standalone vLLM sampling kernels for PTX extraction.
//
// Extracted from vllm/csrc/sampler.cu, stripped of ATen/CUB deps.
// Covers repetition penalties and basic sampling operations.
//
// Compile: nvcc -ptx -arch=sm_80 -O3 sampler_kernels.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

// ============================================================================
// Repetition penalty: penalize tokens that appeared in prompt or output
// ============================================================================

template <typename scalar_t>
__device__ __forceinline__ void apply_repetition_penalties_impl(
    scalar_t* __restrict__ logits,                     // [num_seqs, vocab_size]
    const bool* __restrict__ prompt_mask,              // [num_seqs, vocab_size]
    const bool* __restrict__ output_mask,              // [num_seqs, vocab_size]
    const scalar_t* __restrict__ repetition_penalties,  // [num_seqs]
    const int num_seqs,
    const int vocab_size,
    const int tile_size) {
  const int seq_idx = blockIdx.x;
  if (seq_idx >= num_seqs) return;

  const int tile_start = blockIdx.y * tile_size;
  const int tile_end = min(tile_start + tile_size, vocab_size);

  const scalar_t penalty = repetition_penalties[seq_idx];

  for (int vocab_idx = tile_start + threadIdx.x; vocab_idx < tile_end;
       vocab_idx += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(seq_idx) * vocab_size + vocab_idx;
    const bool is_repeated = prompt_mask[idx] || output_mask[idx];
    if (is_repeated) {
      scalar_t logit = logits[idx];
      if (logit > static_cast<scalar_t>(0)) {
        logits[idx] = logit / penalty;
      } else {
        logits[idx] = logit * penalty;
      }
    }
  }
}

extern "C" __global__
void apply_repetition_penalties_f32(
    float* logits, const bool* prompt_mask, const bool* output_mask,
    const float* repetition_penalties,
    int num_seqs, int vocab_size, int tile_size) {
  apply_repetition_penalties_impl<float>(
      logits, prompt_mask, output_mask, repetition_penalties,
      num_seqs, vocab_size, tile_size);
}

extern "C" __global__
void apply_repetition_penalties_f16(
    __half* logits, const bool* prompt_mask, const bool* output_mask,
    const __half* repetition_penalties,
    int num_seqs, int vocab_size, int tile_size) {
  apply_repetition_penalties_impl<__half>(
      logits, prompt_mask, output_mask, repetition_penalties,
      num_seqs, vocab_size, tile_size);
}

// ============================================================================
// Temperature scaling: divide logits by temperature
// ============================================================================

extern "C" __global__
void temperature_scale_f32(float* __restrict__ logits,
                           const float* __restrict__ temperatures,
                           const int num_seqs,
                           const int vocab_size) {
  const int seq_idx = blockIdx.x;
  if (seq_idx >= num_seqs) return;

  const float temp = temperatures[seq_idx];
  if (temp == 1.0f) return;

  const float inv_temp = 1.0f / temp;
  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(seq_idx) * vocab_size + i;
    logits[idx] *= inv_temp;
  }
}

// ============================================================================
// Top-K masking: set logits outside top-K to -inf
// Uses a simple partial sort approach for small K values.
// For large vocab / K, use the full topk kernel instead.
// ============================================================================

extern "C" __global__
void top_k_mask_f32(float* __restrict__ logits,
                    const int* __restrict__ top_k_values,  // [num_seqs]
                    const int num_seqs,
                    const int vocab_size) {
  const int seq_idx = blockIdx.x;
  if (seq_idx >= num_seqs) return;

  const int k = top_k_values[seq_idx];
  if (k <= 0 || k >= vocab_size) return;

  // Find the k-th largest value using a simple threshold approach
  // This is O(vocab_size) per thread but works for moderate vocab sizes
  float* row = logits + static_cast<int64_t>(seq_idx) * vocab_size;

  // Thread 0 finds the threshold using a simple nth-element approach
  __shared__ float threshold;
  if (threadIdx.x == 0) {
    // Simple approach: find k-th largest by iterating
    // For production, use the radix-based topk kernel instead
    float kth = -FLT_MAX;
    int count = 0;

    // Binary search on threshold value is complex; use iterative approach
    // Start from max and work down
    float lo = -FLT_MAX, hi = FLT_MAX;

    // Find actual max first
    hi = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
      if (row[i] > hi) hi = row[i];
    }
    lo = -FLT_MAX;

    // Binary search for the threshold
    for (int iter = 0; iter < 64; iter++) {
      float mid = lo + (hi - lo) * 0.5f;
      count = 0;
      for (int i = 0; i < vocab_size; i++) {
        if (row[i] > mid) count++;
      }
      if (count > k) {
        lo = mid;
      } else if (count < k) {
        hi = mid;
      } else {
        lo = mid;
        break;
      }
    }
    threshold = lo;
  }
  __syncthreads();

  // Mask logits below threshold
  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(seq_idx) * vocab_size + i;
    if (logits[idx] < threshold) {
      logits[idx] = -FLT_MAX;
    }
  }
}

// ============================================================================
// Softmax for sampling probabilities
// ============================================================================

// Warp reduction helpers
__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

extern "C" __global__
void softmax_f32(float* __restrict__ output,
                 const float* __restrict__ input,
                 const int num_rows,
                 const int num_cols) {
  const int row = blockIdx.x;
  if (row >= num_rows) return;

  const float* in_row = input + static_cast<int64_t>(row) * num_cols;
  float* out_row = output + static_cast<int64_t>(row) * num_cols;

  // Find max (for numerical stability)
  float thread_max = -FLT_MAX;
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    thread_max = fmaxf(thread_max, in_row[i]);
  }

  // Block reduction for max
  __shared__ float shared_max[32];
  float block_max = warp_reduce_max(thread_max);
  if (threadIdx.x % 32 == 0) shared_max[threadIdx.x / 32] = block_max;
  __syncthreads();
  if (threadIdx.x < blockDim.x / 32) block_max = shared_max[threadIdx.x];
  else block_max = -FLT_MAX;
  if (threadIdx.x / 32 == 0) block_max = warp_reduce_max(block_max);
  __shared__ float s_max;
  if (threadIdx.x == 0) s_max = block_max;
  __syncthreads();

  // Compute exp and sum
  float thread_sum = 0.0f;
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    float val = expf(in_row[i] - s_max);
    out_row[i] = val;
    thread_sum += val;
  }

  // Block reduction for sum
  __shared__ float shared_sum[32];
  float block_sum = warp_reduce_sum(thread_sum);
  if (threadIdx.x % 32 == 0) shared_sum[threadIdx.x / 32] = block_sum;
  __syncthreads();
  if (threadIdx.x < blockDim.x / 32) block_sum = shared_sum[threadIdx.x];
  else block_sum = 0.0f;
  if (threadIdx.x / 32 == 0) block_sum = warp_reduce_sum(block_sum);
  __shared__ float s_sum;
  if (threadIdx.x == 0) s_sum = block_sum;
  __syncthreads();

  // Normalize
  float inv_sum = 1.0f / s_sum;
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    out_row[i] *= inv_sum;
  }
}

// ============================================================================
// Multinomial sampling from probabilities
// ============================================================================

extern "C" __global__
void multinomial_sample_f32(
    const float* __restrict__ probs,    // [num_seqs, vocab_size]
    const float* __restrict__ uniform,  // [num_seqs] random values in [0, 1)
    int32_t* __restrict__ output,       // [num_seqs] sampled token ids
    const int num_seqs,
    const int vocab_size) {
  const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq_idx >= num_seqs) return;

  const float* row = probs + static_cast<int64_t>(seq_idx) * vocab_size;
  float u = uniform[seq_idx];

  // CDF scan to find the sampled token
  float cumsum = 0.0f;
  int32_t sampled = vocab_size - 1;  // fallback to last token

  for (int i = 0; i < vocab_size; i++) {
    cumsum += row[i];
    if (cumsum > u) {
      sampled = i;
      break;
    }
  }

  output[seq_idx] = sampled;
}
