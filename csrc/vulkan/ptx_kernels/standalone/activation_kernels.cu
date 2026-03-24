// SPDX-License-Identifier: Apache-2.0
// Standalone vLLM activation kernels for PTX extraction.
//
// These are the raw __global__ kernels from vllm/csrc/activation_kernels.cu,
// stripped of PyTorch/ATen dependencies for compilation to standalone PTX.
// The PTX is loaded via VK_NV_cuda_kernel_launch.
//
// Compile: nvcc -ptx -arch=sm_80 -O3 --use_fast_math activation_kernels.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// ============================================================================
// SiLU (Swish): x * sigmoid(x)
// ============================================================================

extern "C" __global__
void silu_and_mul_f32(float* __restrict__ out,
                      const float* __restrict__ input,
                      const int d) {
  const float* x_ptr = input + blockIdx.x * 2 * d;
  const float* y_ptr = x_ptr + d;
  float* out_ptr = out + blockIdx.x * d;

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = x_ptr[idx];
    float y = y_ptr[idx];
    float silu_x = x / (1.0f + expf(-x));
    out_ptr[idx] = silu_x * y;
  }
}

extern "C" __global__
void silu_and_mul_f16(__half* __restrict__ out,
                      const __half* __restrict__ input,
                      const int d) {
  const __half* x_ptr = input + blockIdx.x * 2 * d;
  const __half* y_ptr = x_ptr + d;
  __half* out_ptr = out + blockIdx.x * d;

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = __half2float(x_ptr[idx]);
    float y = __half2float(y_ptr[idx]);
    float silu_x = x / (1.0f + expf(-x));
    out_ptr[idx] = __float2half(silu_x * y);
  }
}

extern "C" __global__
void silu_and_mul_bf16(__nv_bfloat16* __restrict__ out,
                       const __nv_bfloat16* __restrict__ input,
                       const int d) {
  const __nv_bfloat16* x_ptr = input + blockIdx.x * 2 * d;
  const __nv_bfloat16* y_ptr = x_ptr + d;
  __nv_bfloat16* out_ptr = out + blockIdx.x * d;

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = __bfloat162float(x_ptr[idx]);
    float y = __bfloat162float(y_ptr[idx]);
    float silu_x = x / (1.0f + expf(-x));
    out_ptr[idx] = __float2bfloat16(silu_x * y);
  }
}

// ============================================================================
// GeLU: x * 0.5 * (1 + erf(x / sqrt(2)))
// ============================================================================

extern "C" __global__
void gelu_and_mul_f32(float* __restrict__ out,
                      const float* __restrict__ input,
                      const int d) {
  const float* x_ptr = input + blockIdx.x * 2 * d;
  const float* y_ptr = x_ptr + d;
  float* out_ptr = out + blockIdx.x * d;

  constexpr float ALPHA = 0.7071067811865476f;  // 1/sqrt(2)

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = x_ptr[idx];
    float y = y_ptr[idx];
    float gelu_x = x * 0.5f * (1.0f + erff(x * ALPHA));
    out_ptr[idx] = gelu_x * y;
  }
}

extern "C" __global__
void gelu_and_mul_f16(__half* __restrict__ out,
                      const __half* __restrict__ input,
                      const int d) {
  const __half* x_ptr = input + blockIdx.x * 2 * d;
  const __half* y_ptr = x_ptr + d;
  __half* out_ptr = out + blockIdx.x * d;

  constexpr float ALPHA = 0.7071067811865476f;

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = __half2float(x_ptr[idx]);
    float y = __half2float(y_ptr[idx]);
    float gelu_x = x * 0.5f * (1.0f + erff(x * ALPHA));
    out_ptr[idx] = __float2half(gelu_x * y);
  }
}

extern "C" __global__
void gelu_and_mul_bf16(__nv_bfloat16* __restrict__ out,
                       const __nv_bfloat16* __restrict__ input,
                       const int d) {
  const __nv_bfloat16* x_ptr = input + blockIdx.x * 2 * d;
  const __nv_bfloat16* y_ptr = x_ptr + d;
  __nv_bfloat16* out_ptr = out + blockIdx.x * d;

  constexpr float ALPHA = 0.7071067811865476f;

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = __bfloat162float(x_ptr[idx]);
    float y = __bfloat162float(y_ptr[idx]);
    float gelu_x = x * 0.5f * (1.0f + erff(x * ALPHA));
    out_ptr[idx] = __float2bfloat16(gelu_x * y);
  }
}

// ============================================================================
// GeLU tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// ============================================================================

extern "C" __global__
void gelu_tanh_and_mul_f32(float* __restrict__ out,
                           const float* __restrict__ input,
                           const int d) {
  const float* x_ptr = input + blockIdx.x * 2 * d;
  const float* y_ptr = x_ptr + d;
  float* out_ptr = out + blockIdx.x * d;

  constexpr float BETA = 0.7978845608028654f;  // sqrt(2/pi)
  constexpr float KAPPA = 0.044715f;

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = x_ptr[idx];
    float y = y_ptr[idx];
    float gelu_x = 0.5f * x * (1.0f + tanhf(BETA * x * (1.0f + KAPPA * x * x)));
    out_ptr[idx] = gelu_x * y;
  }
}

extern "C" __global__
void gelu_tanh_and_mul_f16(__half* __restrict__ out,
                           const __half* __restrict__ input,
                           const int d) {
  const __half* x_ptr = input + blockIdx.x * 2 * d;
  const __half* y_ptr = x_ptr + d;
  __half* out_ptr = out + blockIdx.x * d;

  constexpr float BETA = 0.7978845608028654f;
  constexpr float KAPPA = 0.044715f;

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = __half2float(x_ptr[idx]);
    float y = __half2float(y_ptr[idx]);
    float gelu_x = 0.5f * x * (1.0f + tanhf(BETA * x * (1.0f + KAPPA * x * x)));
    out_ptr[idx] = __float2half(gelu_x * y);
  }
}

extern "C" __global__
void gelu_tanh_and_mul_bf16(__nv_bfloat16* __restrict__ out,
                            const __nv_bfloat16* __restrict__ input,
                            const int d) {
  const __nv_bfloat16* x_ptr = input + blockIdx.x * 2 * d;
  const __nv_bfloat16* y_ptr = x_ptr + d;
  __nv_bfloat16* out_ptr = out + blockIdx.x * d;

  constexpr float BETA = 0.7978845608028654f;
  constexpr float KAPPA = 0.044715f;

  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    float x = __bfloat162float(x_ptr[idx]);
    float y = __bfloat162float(y_ptr[idx]);
    float gelu_x = 0.5f * x * (1.0f + tanhf(BETA * x * (1.0f + KAPPA * x * x)));
    out_ptr[idx] = __float2bfloat16(gelu_x * y);
  }
}

// ============================================================================
// Standalone activation (no gating): SiLU, GeLU, ReLU, QuickGeLU
// ============================================================================

extern "C" __global__
void silu_activation_f32(float* __restrict__ out,
                         const float* __restrict__ input,
                         const int d) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < d; idx += blockDim.x * gridDim.x) {
    float x = input[idx];
    out[idx] = x / (1.0f + expf(-x));
  }
}

extern "C" __global__
void gelu_activation_f32(float* __restrict__ out,
                         const float* __restrict__ input,
                         const int d) {
  constexpr float ALPHA = 0.7071067811865476f;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < d; idx += blockDim.x * gridDim.x) {
    float x = input[idx];
    out[idx] = x * 0.5f * (1.0f + erff(x * ALPHA));
  }
}

extern "C" __global__
void relu_activation_f32(float* __restrict__ out,
                         const float* __restrict__ input,
                         const int d) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < d; idx += blockDim.x * gridDim.x) {
    out[idx] = fmaxf(input[idx], 0.0f);
  }
}
