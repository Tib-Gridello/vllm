// SPDX-License-Identifier: Apache-2.0
// Standalone GEMM kernels for PTX extraction.
//
// These are simplified GEMM kernels for use when cuBLAS is not available.
// They use a tiled approach with shared memory for reasonable performance.
//
// For production use, CUTLASS PTX extraction is preferred (same perf as cuBLAS).
// These kernels serve as a fallback when CUTLASS PTX is not pre-compiled.
//
// Performance note: These are ~30-50% of cuBLAS speed for typical LLM shapes.
// CUTLASS PTX extraction (from the 20 CUTLASS .cu files in vLLM) would give
// ~95-100% of cuBLAS speed since it runs the same PTX.
//
// Compile: nvcc -ptx -arch=sm_80 -O3 --use_fast_math gemm_kernels.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

// ============================================================================
// Tiled GEMM: C = alpha * A @ B + beta * C
//
// A: [M, K], B: [K, N], C: [M, N]
// Each thread block computes a TILE_M x TILE_N tile of C.
//
// This is a basic tiled GEMM without advanced optimizations (no double
// buffering, no vectorized loads, no tensor cores). It's a functional
// fallback for when cuBLAS is unavailable.
// ============================================================================

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREADS_X 16
#define THREADS_Y 16

extern "C" __global__
void gemm_nn_f32(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const int M, const int N, const int K,
    const float alpha, const float beta) {

  // Shared memory for tiles of A and B
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];

  const int bx = blockIdx.x;  // tile column
  const int by = blockIdx.y;  // tile row
  const int tx = threadIdx.x; // thread column within tile
  const int ty = threadIdx.y; // thread row within tile

  // Each thread computes a 4x4 sub-tile of C
  const int row0 = by * TILE_M + ty * 4;
  const int col0 = bx * TILE_N + tx * 4;

  float acc[4][4] = {{0.0f}};

  // Loop over tiles of K dimension
  for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    // Cooperative load of A tile
    for (int i = ty; i < TILE_M; i += THREADS_Y) {
      for (int j = tx; j < TILE_K; j += THREADS_X) {
        int row = by * TILE_M + i;
        int col = k_tile + j;
        As[i][j] = (row < M && col < K) ? A[row * K + col] : 0.0f;
      }
    }

    // Cooperative load of B tile
    for (int i = ty; i < TILE_K; i += THREADS_Y) {
      for (int j = tx; j < TILE_N; j += THREADS_X) {
        int row = k_tile + i;
        int col = bx * TILE_N + j;
        Bs[i][j] = (row < K && col < N) ? B[row * N + col] : 0.0f;
      }
    }

    __syncthreads();

    // Compute 4x4 sub-tile
    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      float a_vals[4], b_vals[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        a_vals[i] = As[ty * 4 + i][k];
        b_vals[i] = Bs[k][tx * 4 + i];
      }
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          acc[i][j] += a_vals[i] * b_vals[j];
        }
      }
    }

    __syncthreads();
  }

  // Write results
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int row = row0 + i;
      int col = col0 + j;
      if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = alpha * acc[i][j] + beta * C[idx];
      }
    }
  }
}

// ============================================================================
// f16 GEMM with float accumulation (most common for LLM inference)
// ============================================================================

extern "C" __global__
void gemm_nn_f16(
    __half* __restrict__ C,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const int M, const int N, const int K,
    const float alpha, const float beta) {

  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];

  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row0 = by * TILE_M + ty * 4;
  const int col0 = bx * TILE_N + tx * 4;

  float acc[4][4] = {{0.0f}};

  for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    for (int i = ty; i < TILE_M; i += THREADS_Y) {
      for (int j = tx; j < TILE_K; j += THREADS_X) {
        int row = by * TILE_M + i;
        int col = k_tile + j;
        As[i][j] = (row < M && col < K) ? __half2float(A[row * K + col]) : 0.0f;
      }
    }

    for (int i = ty; i < TILE_K; i += THREADS_Y) {
      for (int j = tx; j < TILE_N; j += THREADS_X) {
        int row = k_tile + i;
        int col = bx * TILE_N + j;
        Bs[i][j] = (row < K && col < N) ? __half2float(B[row * N + col]) : 0.0f;
      }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      float a_vals[4], b_vals[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        a_vals[i] = As[ty * 4 + i][k];
        b_vals[i] = Bs[k][tx * 4 + i];
      }
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          acc[i][j] += a_vals[i] * b_vals[j];
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int row = row0 + i;
      int col = col0 + j;
      if (row < M && col < N) {
        int idx = row * N + col;
        float c_val = beta != 0.0f ? __half2float(C[idx]) : 0.0f;
        C[idx] = __float2half(alpha * acc[i][j] + beta * c_val);
      }
    }
  }
}

// ============================================================================
// Linear layer: output = input @ weight.T + bias
// This is the most common op in transformers (F.linear)
//
// input: [M, K], weight: [N, K] (transposed!), output: [M, N]
// ============================================================================

extern "C" __global__
void linear_f16(
    __half* __restrict__ output,       // [M, N]
    const __half* __restrict__ input,  // [M, K]
    const __half* __restrict__ weight, // [N, K] (row-major, needs transpose)
    const __half* __restrict__ bias,   // [N] or nullptr
    const int M, const int N, const int K) {

  // Simple per-thread implementation for correctness.
  // For performance, use the tiled GEMM above with transposed B.
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k = 0; k < K; k++) {
    // weight is [N, K], so weight[col, k] = weight[col * K + k]
    sum += __half2float(input[row * K + k]) *
           __half2float(weight[col * K + k]);
  }

  if (bias != nullptr) {
    sum += __half2float(bias[col]);
  }

  output[row * N + col] = __float2half(sum);
}

extern "C" __global__
void linear_bf16(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    const int M, const int N, const int K) {

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k = 0; k < K; k++) {
    sum += __bfloat162float(input[row * K + k]) *
           __bfloat162float(weight[col * K + k]);
  }

  if (bias != nullptr) {
    sum += __bfloat162float(bias[col]);
  }

  output[row * N + col] = __float2bfloat16(sum);
}
