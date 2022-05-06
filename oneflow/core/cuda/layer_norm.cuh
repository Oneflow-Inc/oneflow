/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ONEFLOW_CORE_CUDA_LAYER_NORM_H_
#define ONEFLOW_CORE_CUDA_LAYER_NORM_H_

#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>

namespace oneflow {

namespace cuda {

namespace layer_norm {

constexpr int kWarpSize = 32;

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> class ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask, thread_group_width));
  }
  return val;
}

template<template<typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) { result_broadcast = result; }
  __syncthreads();
  return result_broadcast;
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

template<typename T>
__inline__ __device__ T Rsqrt(T x);

template<>
__inline__ __device__ float Rsqrt<float>(float x) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
  return __frsqrt_rn(x);
#else
  return rsqrt(x);
#endif
}

template<>
__inline__ __device__ double Rsqrt<double>(double x) {
  return rsqrt(x);
}

template<class Func>
inline cudaError_t GetNumBlocks(Func func, int64_t block_size, size_t dynamic_smem_size,
                                int64_t max_blocks, int64_t waves, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int max_active_blocks;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, func,
                                                                    block_size, dynamic_smem_size);
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return cudaSuccess;
}

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

#if CUDA_VERSION >= 11000
template<>
struct DefaultComputeType<nv_bfloat16> {
  using type = float;
};
#endif  // CUDA_VERSION >= 11000

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template<typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  const SRC* src;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

template<typename T>
inline __device__ void WelfordCombine(T val, T* mean, T* m2, T* count) {
  // Use Welford Online algorithem to compute mean and variance
  // For more details you can refer to:
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  *count += 1;
  T delta1 = val - *mean;
  *mean += Div(delta1, *count);
  T delta2 = val - *mean;
  *m2 += delta1 * delta2;
}

template<typename T>
inline __device__ void WelfordCombine(T b_mean, T b_m2, T b_count, T* mean, T* m2, T* count) {
  if (b_count == 0) { return; }
  T new_count = *count + b_count;
  T nb_over_n = Div(b_count, new_count);
  T delta = b_mean - *mean;
  *mean += delta * nb_over_n;
  *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
  *count = new_count;
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                             T* m2, T* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                                T* m2, T* count) {
  WelfordWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_count, mean, m2, count);
  *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

template<typename T>
__inline__ __device__ void WelfordBlockAllReduce(T thread_mean, T thread_m2, T thread_count,
                                                 T* result_mean, T* result_m2, T* result_count) {
  __shared__ T mean_shared[kWarpSize];
  __shared__ T m2_shared[kWarpSize];
  __shared__ T count_shared[kWarpSize];
  __shared__ T mean_result_broadcast;
  __shared__ T m2_result_broadcast;
  __shared__ T count_result_broadcast;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  T warp_mean = 0;
  T warp_m2 = 0;
  T warp_count = 0;
  WelfordWarpReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
  __syncthreads();
  if (lid == 0) {
    mean_shared[wid] = warp_mean;
    m2_shared[wid] = warp_m2;
    count_shared[wid] = warp_count;
  }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_mean = mean_shared[lid];
      warp_m2 = m2_shared[lid];
      warp_count = count_shared[lid];
    } else {
      warp_mean = static_cast<T>(0);
      warp_m2 = static_cast<T>(0);
      warp_count = static_cast<T>(0);
    }
    __syncwarp();
    T block_mean = 0;
    T block_m2 = 0;
    T block_count = 0;
    WelfordWarpReduce(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
    if (lid == 0) {
      mean_result_broadcast = block_mean;
      m2_result_broadcast = block_m2;
      count_result_broadcast = block_count;
    }
  }
  __syncthreads();
  *result_mean = mean_result_broadcast;
  *result_m2 = m2_result_broadcast;
  *result_count = count_result_broadcast;
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType* mean,
                                  ComputeType* inv_variance) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int64_t num_global_thread_group = gridDim.x * blockDim.y;
  const int64_t lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_mean[rows_per_access];
    ComputeType thread_m2[rows_per_access];
    ComputeType thread_count[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_mean[row_id] = 0;
      thread_m2[row_id] = 0;
      thread_count[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        const int pack_offset = pack_id * pack_size;
        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            WelfordCombine(row_buf[pack_offset + i], thread_mean + row_id, thread_m2 + row_id,
                           thread_count + row_id);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = 0; }
        }
      }
    }
    ComputeType warp_mean[rows_per_access];
    ComputeType warp_m2[rows_per_access];
    ComputeType warp_count[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      int global_row_id = row + row_id;
      ComputeType* row_buf = buf[row_id];
      WelfordWarpAllReduce<ComputeType, thread_group_width>(
          thread_mean[row_id], thread_m2[row_id], thread_count[row_id], warp_mean + row_id,
          warp_m2 + row_id, warp_count + row_id);
      ComputeType row_mean = warp_mean[row_id];
      ComputeType row_variance =
          max(Div(warp_m2[row_id], warp_count[row_id]), static_cast<ComputeType>(0.0));
      ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
      if (lane_id == 0) {
        mean[global_row_id] = row_mean;
        inv_variance[global_row_id] = row_inv_var;
      }
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = (row_buf[i] - row_mean) * row_inv_var;
      }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col);
        }
      }
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
inline cudaError_t LaunchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols,
                                           const double epsilon, ComputeType* mean,
                                           ComputeType* inv_variance) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                       thread_group_width, rows_per_access, padding>,
                     block_size, 0, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                    rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access>
inline cudaError_t DispatchLayerNormWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols,
                                                    const double epsilon, ComputeType* mean,
                                                    ComputeType* inv_variance) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                   thread_group_width, rows_per_access, false>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  } else {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                   thread_group_width, rows_per_access, true>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}
template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 4, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    if (rows % 2 == 0) {                                                                      \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance) {
    if (cols % 4 == 0) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                             const int64_t rows, const int64_t cols,
                                             const double epsilon, ComputeType* mean,
                                             ComputeType* inv_variance) {
  return DispatchLayerNormWarpImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LayerNormBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                       const int64_t cols, const double epsilon, ComputeType* mean,
                                       ComputeType* inv_variance) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = static_cast<int>(cols) / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        WelfordCombine(pack[i], &thread_mean, &thread_m2, &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_count = 0;
    WelfordBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2,
                                       &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      mean[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = (buf[i * num_packs + pack_id] - row_mean) * row_inv_var;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
inline cudaError_t LaunchLayerNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                int smem, const int64_t rows, const int64_t cols,
                                                const double epsilon, ComputeType* mean,
                                                ComputeType* inv_variance) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>,
                     block_size, smem, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols, epsilon, mean,
                                                 inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline cudaError_t TryDispatchLayerNormBlockSMemImplBlockSize(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType);
  int max_active_blocks_conf_1;

  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  *success = true;
  return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>(
      stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType>
struct TryDispatchLayerNormBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance, bool* success) {
    if (cols % 4 == 0) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    } else if (cols % 2 == 0) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    } else {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t TryDispatchLayerNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                     const int64_t rows, const int64_t cols,
                                                     const double epsilon, ComputeType* mean,
                                                     ComputeType* inv_variance, bool* success) {
  return TryDispatchLayerNormBlockSMemImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LayerNormBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                           const int64_t cols, const double epsilon,
                                           ComputeType* mean, ComputeType* inv_variance) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = static_cast<int>(cols) / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        WelfordCombine(pack[i], &thread_mean, &thread_m2, &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_count = 0;
    WelfordBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2,
                                       &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      mean[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      const int pack_offset = pack_id * pack_size;
      load.template load<pack_size>(pack, row, pack_offset);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { pack[i] = (pack[i] - row_mean) * row_inv_var; }
      store.template store<pack_size>(pack, row, pack_offset);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline cudaError_t LaunchLayerNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols,
                                                    const double epsilon, ComputeType* mean,
                                                    ComputeType* inv_variance) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>,
                     block_size, 0, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance) {
    if (cols % 4 == 0) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                      const int64_t rows, const int64_t cols,
                                                      const double epsilon, ComputeType* mean,
                                                      ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNorm(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols, const double epsilon, ComputeType* mean,
                  ComputeType* inv_variance) {
  if (cols <= 1024) {
    return DispatchLayerNormWarpImpl<LOAD, STORE, ComputeType>(stream, load, store, rows, cols,
                                                               epsilon, mean, inv_variance);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err = TryDispatchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance,
          &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNorm(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols, const double epsilon, ComputeType* mean,
                  ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType>(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

/*
LayerNormGrad dx:
normalized = (x - mean) * inv_var
sum_stats1 = sum(scaled_dy)
sum_stats2 = sum(scaled_dy * normalized)
dx = cols * dy - sum_stats1 - normalized * sum_stats2
dx *= inv_var / cols
*/
template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size, int cols_per_thread, int thread_group_width, int rows_per_access,
         bool padding>
__global__ void LayerNormGradWarpImpl(LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy, STORE store,
                                      const ComputeType* mean, const ComputeType* inv_variance,
                                      const int64_t rows, const int64_t cols) {
  static_assert(cols_per_thread % pack_size == 0, "");
  constexpr int pack_per_thread = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  ComputeType normalized_buf[rows_per_access][cols_per_thread];
  ComputeType dy_buf[rows_per_access][cols_per_thread];
  const ComputeType one_over_cols = static_cast<ComputeType>(1.0) / static_cast<ComputeType>(cols);
  const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int64_t num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType sum_stats1[rows_per_access];
    ComputeType sum_stats2[rows_per_access];
    ComputeType inv_variance_buf[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      const int global_row_id = row + row_id;
      ComputeType mean_val = mean[global_row_id];
      inv_variance_buf[row_id] = inv_variance[global_row_id];
      sum_stats1[row_id] = 0;
      sum_stats2[row_id] = 0;
      ComputeType* row_normalized_buf = normalized_buf[row_id];
      ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        const int pack_offset = pack_id * pack_size;
        if (!padding || col < cols) {
          load_x.template load<pack_size>(row_normalized_buf + pack_offset, global_row_id, col);
          load_scaled_dy.template load<pack_size>(row_dy_buf + pack_offset, global_row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            const int col_id = pack_offset + i;
            // row_normalized_buf store x
            row_normalized_buf[col_id] =
                (row_normalized_buf[col_id] - mean_val) * inv_variance_buf[row_id];
            sum_stats1[row_id] += row_dy_buf[col_id];
            sum_stats2[row_id] += row_dy_buf[col_id] * row_normalized_buf[col_id];
          }
        }
      }
    }
    ComputeType warp_sum_stats1[rows_per_access];
    ComputeType warp_sum_stats2[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum_stats1[row_id] =
          WarpAllReduce<SumOp, ComputeType, thread_group_width>(sum_stats1[row_id]);
      warp_sum_stats2[row_id] =
          WarpAllReduce<SumOp, ComputeType, thread_group_width>(sum_stats2[row_id]);
    }
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      const int global_row_id = row + row_id;
      ComputeType* row_normalized_buf = normalized_buf[row_id];
      ComputeType* row_dy_buf = dy_buf[row_id];
      const ComputeType inv_variance_over_cols = inv_variance_buf[row_id] * one_over_cols;
#pragma unroll
      for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          for (int i = 0; i < pack_size; ++i) {
            const int col_id = pack_id * pack_size + i;
            row_dy_buf[col_id] = (cols * row_dy_buf[col_id] - warp_sum_stats1[row_id]
                                  - row_normalized_buf[col_id] * warp_sum_stats2[row_id])
                                 * inv_variance_over_cols;
          }
          store.template store<pack_size>(row_dy_buf + pack_id * pack_size, global_row_id, col);
        }
      }
    }
  }
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size, int cols_per_thread, int thread_group_width, int rows_per_access,
         bool padding>
inline cudaError_t LaunchLayerNormGradWarpImpl(cudaStream_t stream, LOAD_X load_x,
                                               LOAD_SCALED_DY load_scaled_dy, STORE store,
                                               const ComputeType* mean,
                                               const ComputeType* inv_variance, const int64_t rows,
                                               const int64_t cols) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(
        LayerNormGradWarpImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                              cols_per_thread, thread_group_width, rows_per_access, padding>,
        block_size, 0, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormGradWarpImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size, cols_per_thread,
                        thread_group_width, rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load_x, load_scaled_dy, store, mean, inv_variance,
                                             rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size, int cols_per_thread, int thread_group_width, int rows_per_access>
inline cudaError_t DispatchLayerNormGradWarpImplPadding(cudaStream_t stream, LOAD_X load_x,
                                                        LOAD_SCALED_DY load_scaled_dy, STORE store,
                                                        const ComputeType* mean,
                                                        const ComputeType* inv_variance,
                                                        const int64_t rows, const int64_t cols) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchLayerNormGradWarpImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                       cols_per_thread, thread_group_width, rows_per_access, false>(
        stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
  } else {
    return LaunchLayerNormGradWarpImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                       cols_per_thread, thread_group_width, rows_per_access, true>(
        stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
  }
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLayerNormGradWarpImplCols(
    cudaStream_t stream, LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy, STORE store,
    const ComputeType* mean, const ComputeType* inv_variance, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                     \
  else if (cols <= (thread_group_width)*pack_size) {                                            \
    if (rows % 2 == 0) {                                                                        \
      return DispatchLayerNormGradWarpImplPadding<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType,   \
                                                  pack_size, pack_size, thread_group_width, 2>( \
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);               \
    } else {                                                                                    \
      return DispatchLayerNormGradWarpImplPadding<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType,   \
                                                  pack_size, pack_size, thread_group_width, 1>( \
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);               \
    }                                                                                           \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                \
  else if (cols <= (col)*kWarpSize) {                                                       \
    return DispatchLayerNormGradWarpImplPadding<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, \
                                                pack_size, col, kWarpSize, 1>(              \
        stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);             \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchLayerNormGradWarpImplCols(
    cudaStream_t stream, LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy, STORE store,
    const ComputeType* mean, const ComputeType* inv_variance, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                     \
  else if (cols <= (thread_group_width)*pack_size) {                                            \
    if (rows % 2 == 0) {                                                                        \
      return DispatchLayerNormGradWarpImplPadding<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType,   \
                                                  pack_size, pack_size, thread_group_width, 2>( \
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);               \
    } else {                                                                                    \
      return DispatchLayerNormGradWarpImplPadding<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType,   \
                                                  pack_size, pack_size, thread_group_width, 1>( \
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);               \
    }                                                                                           \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                \
  else if (cols <= (col)*kWarpSize) {                                                       \
    return DispatchLayerNormGradWarpImplPadding<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, \
                                                pack_size, col, kWarpSize, 1>(              \
        stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);             \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType>
struct DispatchLayerNormGradWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy,
                         STORE store, const ComputeType* mean, const ComputeType* inv_variance,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
      return DispatchLayerNormGradWarpImplCols<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, 2>(
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
    } else {
      return DispatchLayerNormGradWarpImplCols<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, 1>(
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
    }
  }
};

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormGradWarpImpl(cudaStream_t stream, LOAD_X load_x,
                                                 LOAD_SCALED_DY load_scaled_dy, STORE store,
                                                 const ComputeType* mean,
                                                 const ComputeType* inv_variance,
                                                 const int64_t rows, const int64_t cols) {
  return DispatchLayerNormGradWarpImplPackSize<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType>()(
      stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size, int block_size>
__global__ void LayerNormGradBlockSMemImpl(LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy,
                                           STORE store, const ComputeType* mean,
                                           const ComputeType* inv_variance, const int64_t rows,
                                           const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
  auto* normalized_buf = reinterpret_cast<ComputeType*>(grad_shared_buf);
  auto* dy_buf = normalized_buf + cols;
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = static_cast<int>(cols) / pack_size;
  const ComputeType one_over_cols = static_cast<ComputeType>(1.0) / static_cast<ComputeType>(cols);
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType sum_stats1 = 0;
    ComputeType sum_stats2 = 0;
    const ComputeType mean_val = mean[row];
    const ComputeType inv_variance_val = inv_variance[row];
    const ComputeType inv_variance_over_cols = inv_variance_val * one_over_cols;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType x_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_x.template load<pack_size>(x_pack, row, pack_id * pack_size);
      load_scaled_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        const int buf_offset = i * num_packs + pack_id;
        ComputeType normalized = (x_pack[i] - mean_val) * inv_variance_val;
        normalized_buf[buf_offset] = normalized;
        dy_buf[buf_offset] = dy_pack[i];
        sum_stats1 += dy_pack[i];
        sum_stats2 += dy_pack[i] * normalized;
      }
    }
    const ComputeType row_sum_stats1 = BlockAllReduce<SumOp, ComputeType, block_size>(sum_stats1);
    const ComputeType row_sum_stats2 = BlockAllReduce<SumOp, ComputeType, block_size>(sum_stats2);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        const int buf_offset = i * num_packs + pack_id;
        pack[i] = (cols * dy_buf[buf_offset] - row_sum_stats1
                   - normalized_buf[buf_offset] * row_sum_stats2)
                  * inv_variance_over_cols;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size, int block_size>
inline cudaError_t LaunchLayerNormGradBlockSMemImpl(cudaStream_t stream, LOAD_X load_x,
                                                    LOAD_SCALED_DY load_scaled_dy, STORE store,
                                                    const ComputeType* mean,
                                                    const ComputeType* inv_variance, int smem,
                                                    const int64_t rows, const int64_t cols) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(LayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE,
                                                              ComputeType, pack_size, block_size>,
                                   block_size, smem, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(load_x, load_scaled_dy, store, mean, inv_variance,
                                                 rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size>
inline cudaError_t TryDispatchLayerNormGradBlockSMemImplBlockSize(
    cudaStream_t stream, LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy, STORE store,
    const ComputeType* mean, const ComputeType* inv_variance, const int64_t rows,
    const int64_t cols, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType) * 2;
  int max_active_blocks_conf_1;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        LayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                   block_size_conf_1>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        LayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                   block_size_conf_4>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                            block_size_conf_4>(
        stream, load_x, load_scaled_dy, store, mean, inv_variance, smem, rows, cols);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        LayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                   block_size_conf_3>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                            block_size_conf_3>(
        stream, load_x, load_scaled_dy, store, mean, inv_variance, smem, rows, cols);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        LayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                   block_size_conf_2>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                            block_size_conf_2>(
        stream, load_x, load_scaled_dy, store, mean, inv_variance, smem, rows, cols);
  }
  *success = true;
  return LaunchLayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size,
                                          block_size_conf_1>(stream, load_x, load_scaled_dy, store,
                                                             mean, inv_variance, smem, rows, cols);
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType>
struct TryDispatchLayerNormGradBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy,
                         STORE store, const ComputeType* mean, const ComputeType* inv_variance,
                         const int64_t rows, const int64_t cols, bool* success) {
    if (cols % 2 == 0) {
      return TryDispatchLayerNormGradBlockSMemImplBlockSize<LOAD_X, LOAD_SCALED_DY, STORE,
                                                            ComputeType, 2>(
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols, success);
    } else {
      return TryDispatchLayerNormGradBlockSMemImplBlockSize<LOAD_X, LOAD_SCALED_DY, STORE,
                                                            ComputeType, 1>(
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols, success);
    }
  }
};

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType>
inline cudaError_t TryDispatchLayerNormGradBlockSMemImpl(cudaStream_t stream, LOAD_X load_x,
                                                         LOAD_SCALED_DY load_scaled_dy, STORE store,
                                                         const ComputeType* mean,
                                                         const ComputeType* inv_variance,
                                                         const int64_t rows, const int64_t cols,
                                                         bool* success) {
  return TryDispatchLayerNormGradBlockSMemImplPackSize<LOAD_X, LOAD_SCALED_DY, STORE,
                                                       ComputeType>()(
      stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols, success);
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size, int block_size>
__global__ void LayerNormGradBlockUncachedImpl(LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy,
                                               STORE store, const ComputeType* mean,
                                               const ComputeType* inv_variance, const int64_t rows,
                                               const int64_t cols) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = static_cast<int>(cols) / pack_size;
  const ComputeType one_over_cols = static_cast<ComputeType>(1.0) / static_cast<ComputeType>(cols);
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const ComputeType mean_val = mean[row];
    const ComputeType inv_variance_val = inv_variance[row];
    const ComputeType inv_variance_over_cols = inv_variance_val * one_over_cols;
    ComputeType sum_stats1 = 0;
    ComputeType sum_stats2 = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType x_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_x.template load<pack_size>(x_pack, row, pack_id * pack_size);
      load_scaled_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);

#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        sum_stats1 += dy_pack[i];
        sum_stats2 += dy_pack[i] * (x_pack[i] - mean_val) * inv_variance_val;
      }
    }
    const ComputeType row_sum_stats1 = BlockAllReduce<SumOp, ComputeType, block_size>(sum_stats1);
    const ComputeType row_sum_stats2 = BlockAllReduce<SumOp, ComputeType, block_size>(sum_stats2);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType x_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_x.template load<pack_size>(x_pack, row, pack_id * pack_size);
      load_scaled_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        dy_pack[i] = (cols * dy_pack[i] - row_sum_stats1
                      - (x_pack[i] - mean_val) * inv_variance_val * row_sum_stats2)
                     * inv_variance_over_cols;
      }
      store.template store<pack_size>(dy_pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType,
         int pack_size>
inline cudaError_t LaunchLayerNormGradBlockUncachedImpl(cudaStream_t stream, LOAD_X load_x,
                                                        LOAD_SCALED_DY load_scaled_dy, STORE store,
                                                        const ComputeType* mean,
                                                        const ComputeType* inv_variance,
                                                        const int64_t rows, const int64_t cols) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormGradBlockUncachedImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType,
                                                    pack_size, block_size>,
                     block_size, 0, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormGradBlockUncachedImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(load_x, load_scaled_dy, store, mean, inv_variance,
                                              rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType>
struct DispatchLayerNormGradBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy,
                         STORE store, const ComputeType* mean, const ComputeType* inv_variance,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0 && cols > kWarpSize) {
      return LaunchLayerNormGradBlockUncachedImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, 2>(
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
    } else {
      return LaunchLayerNormGradBlockUncachedImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType, 1>(
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
    }
  }
};

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormGradBlockUncachedImpl(cudaStream_t stream, LOAD_X load_x,
                                                          LOAD_SCALED_DY load_scaled_dy,
                                                          STORE store, const ComputeType* mean,
                                                          const ComputeType* inv_variance,
                                                          const int64_t rows, const int64_t cols) {
  return DispatchLayerNormGradBlockUncachedImplPackSize<LOAD_X, LOAD_SCALED_DY, STORE,
                                                        ComputeType>()(
      stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNormGrad(cudaStream_t stream, LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy,
                      STORE store, const ComputeType* mean, const ComputeType* inv_variance,
                      const int64_t rows, const int64_t cols) {
  if (cols <= 1024) {
    return DispatchLayerNormGradWarpImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType>(
        stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err =
          TryDispatchLayerNormGradBlockSMemImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType>(
              stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols,
              &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchLayerNormGradBlockUncachedImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType>(
          stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD_X, typename LOAD_SCALED_DY, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNormGrad(cudaStream_t stream, LOAD_X load_x, LOAD_SCALED_DY load_scaled_dy,
                      STORE store, const ComputeType* mean, const ComputeType* inv_variance,
                      const int64_t rows, const int64_t cols) {
  return DispatchLayerNormGradBlockUncachedImpl<LOAD_X, LOAD_SCALED_DY, STORE, ComputeType>(
      stream, load_x, load_scaled_dy, store, mean, inv_variance, rows, cols);
}

}  // namespace layer_norm

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_LAYER_NORM_H_
