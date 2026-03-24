// SPDX-License-Identifier: Apache-2.0
// Standalone vLLM RMS LayerNorm kernels for PTX extraction.
//
// Stripped of PyTorch/CUB dependencies. Uses warp-level reduction.
// Compile: nvcc -ptx -arch=sm_80 -O3 --use_fast_math layernorm_kernels.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block-level reduction for sum using shared memory
__device__ float block_reduce_sum(float val) {
  __shared__ float shared[32];  // one per warp

  int lane = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  val = warp_reduce_sum(val);

  if (lane == 0) shared[warp_id] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
  if (warp_id == 0) val = warp_reduce_sum(val);

  return val;
}

// ============================================================================
// RMS Normalization: out = (x / sqrt(mean(x^2) + eps)) * weight
// ============================================================================

extern "C" __global__
void rms_norm_f32(float* __restrict__ out,
                  const float* __restrict__ input,
                  const float* __restrict__ weight,
                  const float epsilon,
                  const int hidden_size) {
  const int row = blockIdx.x;
  const float* x = input + row * hidden_size;
  float* o = out + row * hidden_size;

  // Compute variance (mean of squares)
  float variance = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = x[i];
    variance += val * val;
  }
  variance = block_reduce_sum(variance);
  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf(variance / (float)hidden_size + epsilon);
  }
  __syncthreads();

  // Apply normalization and scale
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    o[i] = x[i] * s_inv_rms * weight[i];
  }
}

extern "C" __global__
void rms_norm_f16(__half* __restrict__ out,
                  const __half* __restrict__ input,
                  const __half* __restrict__ weight,
                  const float epsilon,
                  const int hidden_size) {
  const int row = blockIdx.x;
  const __half* x = input + row * hidden_size;
  __half* o = out + row * hidden_size;

  float variance = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = __half2float(x[i]);
    variance += val * val;
  }
  variance = block_reduce_sum(variance);
  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf(variance / (float)hidden_size + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = __half2float(x[i]) * s_inv_rms * __half2float(weight[i]);
    o[i] = __float2half(val);
  }
}

extern "C" __global__
void rms_norm_bf16(__nv_bfloat16* __restrict__ out,
                   const __nv_bfloat16* __restrict__ input,
                   const __nv_bfloat16* __restrict__ weight,
                   const float epsilon,
                   const int hidden_size) {
  const int row = blockIdx.x;
  const __nv_bfloat16* x = input + row * hidden_size;
  __nv_bfloat16* o = out + row * hidden_size;

  float variance = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = __bfloat162float(x[i]);
    variance += val * val;
  }
  variance = block_reduce_sum(variance);
  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf(variance / (float)hidden_size + epsilon);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = __bfloat162float(x[i]) * s_inv_rms * __bfloat162float(weight[i]);
    o[i] = __float2bfloat16(val);
  }
}

// ============================================================================
// Fused Add + RMS Norm: residual += input; out = rms_norm(residual) * weight
// ============================================================================

extern "C" __global__
void fused_add_rms_norm_f32(float* __restrict__ input,
                            float* __restrict__ residual,
                            const float* __restrict__ weight,
                            const float epsilon,
                            const int hidden_size) {
  const int row = blockIdx.x;
  float* x = input + row * hidden_size;
  float* r = residual + row * hidden_size;

  // Add residual and compute variance
  float variance = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = x[i] + r[i];
    r[i] = val;  // Store sum back to residual
    variance += val * val;
  }
  variance = block_reduce_sum(variance);
  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf(variance / (float)hidden_size + epsilon);
  }
  __syncthreads();

  // Normalize and scale
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    x[i] = r[i] * s_inv_rms * weight[i];
  }
}
