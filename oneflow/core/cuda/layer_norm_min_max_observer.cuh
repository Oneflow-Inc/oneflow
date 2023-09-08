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

#ifndef ONEFLOW_CORE_CUDA_LAYER_NORM_MIN_MAX_OBSERVER_H_
#define ONEFLOW_CORE_CUDA_LAYER_NORM_MIN_MAX_OBSERVER_H_

#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>

#include "oneflow/core/cuda/layer_norm.cuh"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/kernel/util/numeric_limits.cuh"

namespace oneflow {

namespace cuda {

namespace layer_norm {

template<typename T>
inline __device__ void WelfordMinMaxCombine(T val, T* mean, T* m2, T* min, T* max, T* count) {
  // Use Welford Online algorithem to compute mean and variance
  // For more details you can refer to:
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  *count += 1;
  T delta1 = val - *mean;
  *mean += Div(delta1, *count);
  T delta2 = val - *mean;
  *m2 += delta1 * delta2;
  *min = BinaryFuncMin<T>::Invoke(val, *min);
  *max = BinaryFuncMax<T>::Invoke(val, *max);
}

template<typename T>
inline __device__ void WelfordMinMaxCombine(T b_mean, T b_m2, T b_min, T b_max, T b_count, T* mean,
                                            T* m2, T* min, T* max, T* count) {
  if (b_count == 0) { return; }
  T new_count = *count + b_count;
  T nb_over_n = Div(b_count, new_count);
  T delta = b_mean - *mean;
  *mean += delta * nb_over_n;
  *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
  *count = new_count;
  *min = BinaryFuncMin<T>::Invoke(b_min, *min);
  *max = BinaryFuncMax<T>::Invoke(b_max, *max);
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordMinMaxWarpReduce(T thread_mean, T thread_m2, T thread_min,
                                                   T thread_max, T thread_count, T* mean, T* m2,
                                                   T* min, T* max, T* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  *min = thread_min;
  *max = thread_max;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
    T b_min = __shfl_down_sync(0xffffffff, *min, mask, thread_group_width);
    T b_max = __shfl_down_sync(0xffffffff, *max, mask, thread_group_width);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
    WelfordMinMaxCombine(b_mean, b_m2, b_min, b_max, b_count, mean, m2, min, max, count);
  }
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordMinMaxWarpAllReduce(T thread_mean, T thread_m2, T thread_min,
                                                      T thread_max, T thread_count, T* mean, T* m2,
                                                      T* min, T* max, T* count) {
  WelfordMinMaxWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_min, thread_max,
                                                 thread_count, mean, m2, min, max, count);
  *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
  *min = __shfl_sync(0xffffffff, *min, 0, thread_group_width);
  *max = __shfl_sync(0xffffffff, *max, 0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

template<typename T>
__inline__ __device__ void WelfordMinMaxBlockAllReduce(T thread_mean, T thread_m2, T thread_min,
                                                       T thread_max, T thread_count, T* result_mean,
                                                       T* result_m2, T* result_min, T* result_max,
                                                       T* result_count) {
  __shared__ T mean_shared[kWarpSize];
  __shared__ T m2_shared[kWarpSize];
  __shared__ T min_shared[kWarpSize];
  __shared__ T max_shared[kWarpSize];
  __shared__ T count_shared[kWarpSize];
  __shared__ T mean_result_broadcast;
  __shared__ T m2_result_broadcast;
  __shared__ T min_result_broadcast;
  __shared__ T max_result_broadcast;
  __shared__ T count_result_broadcast;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  T warp_mean = 0;
  T warp_m2 = 0;
  T warp_min = detail::numeric_limits<T>::max();
  T warp_max = detail::numeric_limits<T>::lowest();
  T warp_count = 0;
  WelfordMinMaxWarpReduce(thread_mean, thread_m2, thread_min, thread_max, thread_count, &warp_mean,
                          &warp_m2, &warp_min, &warp_max, &warp_count);
  __syncthreads();
  if (lid == 0) {
    mean_shared[wid] = warp_mean;
    m2_shared[wid] = warp_m2;
    min_shared[wid] = warp_min;
    max_shared[wid] = warp_max;
    count_shared[wid] = warp_count;
  }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_mean = mean_shared[lid];
      warp_m2 = m2_shared[lid];
      warp_min = min_shared[lid];
      warp_max = max_shared[lid];
      warp_count = count_shared[lid];
    } else {
      warp_mean = static_cast<T>(0);
      warp_m2 = static_cast<T>(0);
      warp_min = detail::numeric_limits<T>::max();
      warp_max = detail::numeric_limits<T>::lowest();
      warp_count = static_cast<T>(0);
    }
    __syncwarp();
    T block_mean = 0;
    T block_m2 = 0;
    T block_min = detail::numeric_limits<T>::max();
    T block_max = detail::numeric_limits<T>::lowest();
    T block_count = 0;
    WelfordMinMaxWarpReduce(warp_mean, warp_m2, warp_min, warp_max, warp_count, &block_mean,
                            &block_m2, &block_min, &block_max, &block_count);
    if (lid == 0) {
      mean_result_broadcast = block_mean;
      m2_result_broadcast = block_m2;
      min_result_broadcast = block_min;
      max_result_broadcast = block_max;
      count_result_broadcast = block_count;
    }
  }
  __syncthreads();
  *result_mean = mean_result_broadcast;
  *result_m2 = m2_result_broadcast;
  *result_min = min_result_broadcast;
  *result_max = max_result_broadcast;
  *result_count = count_result_broadcast;
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access, bool padding>
__global__ void LayerNormMinMaxObserverWarpImpl(LOAD load, STORE store, const int64_t rows,
                                                const int64_t cols, const double epsilon,
                                                T* min_max) {
  using LoadType = typename LOAD::LoadType;
  static_assert(max_cols_per_thread % pack_size == 0, "");
  static_assert(min_cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int max_num_packs = max_cols_per_thread / pack_size;
  constexpr int min_num_packs = min_cols_per_thread / pack_size;
  assert(cols <= max_cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][max_cols_per_thread];
  const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int64_t num_global_thread_group = gridDim.x * blockDim.y;
  const int64_t lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_mean[rows_per_access];
    ComputeType thread_m2[rows_per_access];
    ComputeType thread_min[rows_per_access];
    ComputeType thread_max[rows_per_access];
    ComputeType thread_count[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_mean[row_id] = 0;
      thread_m2[row_id] = 0;
      thread_min[row_id] = detail::numeric_limits<T>::max();
      thread_max[row_id] = detail::numeric_limits<T>::lowest();
      thread_count[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < min_num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        const int pack_offset = pack_id * pack_size;
        LoadType pack[pack_size];
        load.template load<pack_size>(pack, row + row_id, col);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          row_buf[pack_offset + i] = static_cast<ComputeType>(pack[i]);
          WelfordMinMaxCombine(row_buf[pack_offset + i], thread_mean + row_id, thread_m2 + row_id,
                               thread_min + row_id, thread_max + row_id, thread_count + row_id);
        }
      }
      for (int pack_id = min_num_packs; pack_id < max_num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        const int pack_offset = pack_id * pack_size;
        if (!padding || col < cols) {
          LoadType pack[pack_size];
          load.template load<pack_size>(pack, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            row_buf[pack_offset + i] = static_cast<ComputeType>(pack[i]);
            WelfordMinMaxCombine(row_buf[pack_offset + i], thread_mean + row_id, thread_m2 + row_id,
                                 thread_min + row_id, thread_max + row_id, thread_count + row_id);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = 0; }
        }
      }
    }
    ComputeType warp_mean[rows_per_access];
    ComputeType warp_m2[rows_per_access];
    ComputeType warp_min[rows_per_access];
    ComputeType warp_max[rows_per_access];
    ComputeType warp_count[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      int global_row_id = row + row_id;
      ComputeType* row_buf = buf[row_id];
      WelfordMinMaxWarpAllReduce<ComputeType, thread_group_width>(
          thread_mean[row_id], thread_m2[row_id], thread_min[row_id], thread_max[row_id],
          thread_count[row_id], warp_mean + row_id, warp_m2 + row_id, warp_min + row_id,
          warp_max + row_id, warp_count + row_id);
      ComputeType row_mean = warp_mean[row_id];
      ComputeType row_variance =
          max(Div(warp_m2[row_id], warp_count[row_id]), static_cast<ComputeType>(0.0));
      ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
      if (lane_id == 0) {
        min_max[global_row_id << 1] = (warp_min[row_id] - row_mean) * row_inv_var;
        min_max[(global_row_id << 1) + 1] = (warp_max[row_id] - row_mean) * row_inv_var;
      }
#pragma unroll
      for (int i = 0; i < max_cols_per_thread; ++i) {
        row_buf[i] = (row_buf[i] - row_mean) * row_inv_var;
      }
#pragma unroll
      for (int i = 0; i < min_num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col);
      }
#pragma unroll
      for (int i = min_num_packs; i < max_num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col);
        }
      }
    }
  }
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access, bool padding>
inline cudaError_t LaunchLayerNormMinMaxObserverWarpImpl(cudaStream_t stream, LOAD load,
                                                         STORE store, const int64_t rows,
                                                         const int64_t cols, const double epsilon,
                                                         T* min_max) {
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
        GetNumBlocks(LayerNormMinMaxObserverWarpImpl<LOAD, STORE, T, ComputeType, pack_size,
                                                     max_cols_per_thread, min_cols_per_thread,
                                                     thread_group_width, rows_per_access, padding>,
                     block_size, 0, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormMinMaxObserverWarpImpl<LOAD, STORE, T, ComputeType, pack_size, max_cols_per_thread,
                                  min_cols_per_thread, thread_group_width, rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols, epsilon, min_max);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access>
inline cudaError_t DispatchLayerNormMinMaxObserverWarpImplPadding(cudaStream_t stream, LOAD load,
                                                                  STORE store, const int64_t rows,
                                                                  const int64_t cols,
                                                                  const double epsilon,
                                                                  T* min_max) {
  if (cols == max_cols_per_thread * thread_group_width) {
    // when not padding, min_cols_per_thread must equals to max_cols_per_thread, pass
    // max_cols_per_thread as min_cols_per_thread and max_cols_per_thread param.
    return LaunchLayerNormMinMaxObserverWarpImpl<LOAD, STORE, T, ComputeType, pack_size,
                                                 max_cols_per_thread, max_cols_per_thread,
                                                 thread_group_width, rows_per_access, false>(
        stream, load, store, rows, cols, epsilon, min_max);
  } else {
    return LaunchLayerNormMinMaxObserverWarpImpl<LOAD, STORE, T, ComputeType, pack_size,
                                                 max_cols_per_thread, min_cols_per_thread,
                                                 thread_group_width, rows_per_access, true>(
        stream, load, store, rows, cols, epsilon, min_max);
  }
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type
DispatchLayerNormMinMaxObserverWarpImplCols(cudaStream_t stream, LOAD load, STORE store,
                                            const int64_t rows, const int64_t cols,
                                            const double epsilon, T* min_max) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                             \
  else if (cols <= (thread_group_width)*pack_size) {                                    \
    if (rows % 2 == 0) {                                                                \
      return DispatchLayerNormMinMaxObserverWarpImplPadding<                            \
          LOAD, STORE, T, ComputeType, pack_size, pack_size, 0, thread_group_width, 2>( \
          stream, load, store, rows, cols, epsilon, min_max);                           \
    } else {                                                                            \
      return DispatchLayerNormMinMaxObserverWarpImplPadding<                            \
          LOAD, STORE, T, ComputeType, pack_size, pack_size, 0, thread_group_width, 1>( \
          stream, load, store, rows, cols, epsilon, min_max);                           \
    }                                                                                   \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col)                                                         \
  else if (cols <= (max_col)*kWarpSize) {                                                         \
    return DispatchLayerNormMinMaxObserverWarpImplPadding<LOAD, STORE, T, ComputeType, pack_size, \
                                                          max_col, min_col, kWarpSize, 1>(        \
        stream, load, store, rows, cols, epsilon, min_max);                                       \
  }
  DEFINE_ONE_ELIF(2, 1)
  DEFINE_ONE_ELIF(4, 2)
  DEFINE_ONE_ELIF(8, 4)
  DEFINE_ONE_ELIF(12, 8)
  DEFINE_ONE_ELIF(16, 12)
  DEFINE_ONE_ELIF(20, 16)
  DEFINE_ONE_ELIF(24, 20)
  DEFINE_ONE_ELIF(28, 24)
  DEFINE_ONE_ELIF(32, 28)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type
DispatchLayerNormMinMaxObserverWarpImplCols(cudaStream_t stream, LOAD load, STORE store,
                                            const int64_t rows, const int64_t cols,
                                            const double epsilon, T* min_max) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                             \
  else if (cols <= (thread_group_width)*pack_size) {                                    \
    if (rows % 2 == 0) {                                                                \
      return DispatchLayerNormMinMaxObserverWarpImplPadding<                            \
          LOAD, STORE, T, ComputeType, pack_size, pack_size, 0, thread_group_width, 2>( \
          stream, load, store, rows, cols, epsilon, min_max);                           \
    } else {                                                                            \
      return DispatchLayerNormMinMaxObserverWarpImplPadding<                            \
          LOAD, STORE, T, ComputeType, pack_size, pack_size, 0, thread_group_width, 1>( \
          stream, load, store, rows, cols, epsilon, min_max);                           \
    }                                                                                   \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col)                                                         \
  else if ((cols <= (max_col)*kWarpSize) && (cols > (min_col)*kWarpSize)) {                       \
    return DispatchLayerNormMinMaxObserverWarpImplPadding<LOAD, STORE, T, ComputeType, pack_size, \
                                                          max_col, min_col, kWarpSize, 1>(        \
        stream, load, store, rows, cols, epsilon, min_max);                                       \
  }
  DEFINE_ONE_ELIF(4, 2)
  DEFINE_ONE_ELIF(8, 4)
  DEFINE_ONE_ELIF(12, 8)
  DEFINE_ONE_ELIF(16, 12)
  DEFINE_ONE_ELIF(20, 16)
  DEFINE_ONE_ELIF(24, 20)
  DEFINE_ONE_ELIF(28, 24)
  DEFINE_ONE_ELIF(32, 28)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename T, typename ComputeType>
struct DispatchLayerNormMinMaxObserverWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, T* min_max) {
    if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) && CanPackAs<STORE>(store, 2)) {
      return DispatchLayerNormMinMaxObserverWarpImplCols<LOAD, STORE, T, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, min_max);
    } else {
      return DispatchLayerNormMinMaxObserverWarpImplCols<LOAD, STORE, T, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, min_max);
    }
  }
};

template<typename LOAD, typename STORE, typename T, typename ComputeType>
inline cudaError_t DispatchLayerNormMinMaxObserverWarpImpl(cudaStream_t stream, LOAD load,
                                                           STORE store, const int64_t rows,
                                                           const int64_t cols, const double epsilon,
                                                           T* min_max) {
  return DispatchLayerNormMinMaxObserverWarpImplPackSize<LOAD, STORE, T, ComputeType>()(
      stream, load, store, rows, cols, epsilon, min_max);
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size,
         int block_size>
__global__ void LayerNormMinMaxObserverBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                                     const int64_t cols, const double epsilon,
                                                     T* min_max) {
  using LoadType = typename LOAD::LoadType;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<LoadType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = static_cast<int>(cols) / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_min = detail::numeric_limits<T>::max();
    ComputeType thread_max = detail::numeric_limits<T>::lowest();
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        WelfordMinMaxCombine(static_cast<ComputeType>(pack[i]), &thread_mean, &thread_m2,
                             &thread_min, &thread_max, &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_min = detail::numeric_limits<T>::max();
    ComputeType row_max = detail::numeric_limits<T>::lowest();
    ComputeType row_count = 0;
    WelfordMinMaxBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_min, thread_max,
                                             thread_count, &row_mean, &row_m2, &row_min, &row_max,
                                             &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      min_max[row << 1] = (row_min - row_mean) * row_inv_var;
      min_max[(row << 1) + 1] = (row_max - row_mean) * row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = (static_cast<ComputeType>(buf[i * num_packs + pack_id]) - row_mean) * row_inv_var;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size,
         int block_size>
inline cudaError_t LaunchLayerNormMinMaxObserverBlockSMemImpl(cudaStream_t stream, LOAD load,
                                                              STORE store, int smem,
                                                              const int64_t rows,
                                                              const int64_t cols,
                                                              const double epsilon, T* min_max) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size, block_size>,
        block_size, smem, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols, epsilon, min_max);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size>
inline cudaError_t TryDispatchLayerNormMinMaxObserverBlockSMemImplBlockSize(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, T* min_max, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;

  int dev = 0;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }

  int sm_count = 0;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }

  static const bool max_smem_configed = [=]() {
    int max_smem_size = 0;
    cudaError_t err =
        cudaDeviceGetAttribute(&max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (err != cudaSuccess) { return false; }

    err = MaximizeDynamicSharedMemorySize(
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                             block_size_conf_1>,
        max_smem_size);
    if (err != cudaSuccess) { return false; }
    err = MaximizeDynamicSharedMemorySize(
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                             block_size_conf_2>,
        max_smem_size);
    if (err != cudaSuccess) { return false; }
    err = MaximizeDynamicSharedMemorySize(
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                             block_size_conf_3>,
        max_smem_size);
    if (err != cudaSuccess) { return false; }
    err = MaximizeDynamicSharedMemorySize(
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                             block_size_conf_4>,
        max_smem_size);
    if (err != cudaSuccess) { return false; }

    return true;
  }();

  const size_t smem = cols * sizeof(typename LOAD::LoadType);

  int max_active_blocks_conf_1;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
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
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                             block_size_conf_4>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_4 == max_active_blocks_conf_1
      || (max_active_blocks_conf_4 > 0 && rows <= sm_count)) {
    *success = true;
    return LaunchLayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                                      block_size_conf_4>(
        stream, load, store, smem, rows, cols, epsilon, min_max);
  }

  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                             block_size_conf_3>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1
      || (max_active_blocks_conf_3 > 0 && rows <= sm_count)) {
    *success = true;
    return LaunchLayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                                      block_size_conf_3>(
        stream, load, store, smem, rows, cols, epsilon, min_max);
  }

  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        LayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                             block_size_conf_2>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1
      || (max_active_blocks_conf_2 > 0 && rows <= sm_count)) {
    *success = true;
    return LaunchLayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                                      block_size_conf_2>(
        stream, load, store, smem, rows, cols, epsilon, min_max);
  }

  *success = true;
  return LaunchLayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType, pack_size,
                                                    block_size_conf_1>(
      stream, load, store, smem, rows, cols, epsilon, min_max);
}

template<typename LOAD, typename STORE, typename T, typename ComputeType>
struct TryDispatchLayerNormMinMaxObserverBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, T* min_max, bool* success) {
    if (cols % 4 == 0 && CanPackAs<LOAD>(load, 4) && CanPackAs<STORE>(store, 4)) {
      return TryDispatchLayerNormMinMaxObserverBlockSMemImplBlockSize<LOAD, STORE, T, ComputeType,
                                                                      4>(
          stream, load, store, rows, cols, epsilon, min_max, success);
    } else if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) && CanPackAs<STORE>(store, 2)) {
      return TryDispatchLayerNormMinMaxObserverBlockSMemImplBlockSize<LOAD, STORE, T, ComputeType,
                                                                      2>(
          stream, load, store, rows, cols, epsilon, min_max, success);
    } else {
      return TryDispatchLayerNormMinMaxObserverBlockSMemImplBlockSize<LOAD, STORE, T, ComputeType,
                                                                      1>(
          stream, load, store, rows, cols, epsilon, min_max, success);
    }
  }
};

template<typename LOAD, typename STORE, typename T, typename ComputeType>
inline cudaError_t TryDispatchLayerNormMinMaxObserverBlockSMemImpl(cudaStream_t stream, LOAD load,
                                                                   STORE store, const int64_t rows,
                                                                   const int64_t cols,
                                                                   const double epsilon, T* min_max,
                                                                   bool* success) {
  return TryDispatchLayerNormMinMaxObserverBlockSMemImplPackSize<LOAD, STORE, T, ComputeType>()(
      stream, load, store, rows, cols, epsilon, min_max, success);
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size,
         int block_size>
__global__ void __launch_bounds__(1024)
    LayerNormMinMaxObserverBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                             const int64_t cols, const double epsilon, T* min_max) {
  using LoadType = typename LOAD::LoadType;
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = static_cast<int>(cols) / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_min = detail::numeric_limits<T>::max();
    ComputeType thread_max = detail::numeric_limits<T>::lowest();
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        WelfordMinMaxCombine(static_cast<ComputeType>(pack[i]), &thread_mean, &thread_m2,
                             &thread_min, &thread_max, &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_min = detail::numeric_limits<T>::max();
    ComputeType row_max = detail::numeric_limits<T>::lowest();
    ComputeType row_count = 0;
    WelfordMinMaxBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_min, thread_max,
                                             thread_count, &row_mean, &row_m2, &row_min, &row_max,
                                             &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      min_max[row << 1] = (row_min - row_mean) * row_inv_var;
      min_max[(row << 1) + 1] = (row_max - row_mean) * row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[pack_size];
      ComputeType dst_pack[pack_size];
      const int pack_offset = pack_id * pack_size;
      load.template load<pack_size>(pack, row, pack_offset);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        dst_pack[i] = (static_cast<ComputeType>(pack[i]) - row_mean) * row_inv_var;
      }
      store.template store<pack_size>(dst_pack, row, pack_offset);
    }
  }
}

template<typename LOAD, typename STORE, typename T, typename ComputeType, int pack_size>
inline cudaError_t LaunchLayerNormMinMaxObserverBlockUncachedImpl(cudaStream_t stream, LOAD load,
                                                                  STORE store, const int64_t rows,
                                                                  const int64_t cols,
                                                                  const double epsilon,
                                                                  T* min_max) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormMinMaxObserverBlockUncachedImpl<LOAD, STORE, T, ComputeType,
                                                              pack_size, block_size>,
                     block_size, 0, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormMinMaxObserverBlockUncachedImpl<LOAD, STORE, T, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols, epsilon, min_max);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename T, typename ComputeType>
struct DispatchLayerNormMinMaxObserverBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, T* min_max) {
    if (cols % 4 == 0 && CanPackAs<LOAD>(load, 4) && CanPackAs<STORE>(store, 4)) {
      return LaunchLayerNormMinMaxObserverBlockUncachedImpl<LOAD, STORE, T, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, min_max);
    } else if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) && CanPackAs<STORE>(store, 2)) {
      return LaunchLayerNormMinMaxObserverBlockUncachedImpl<LOAD, STORE, T, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, min_max);
    } else {
      return LaunchLayerNormMinMaxObserverBlockUncachedImpl<LOAD, STORE, T, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, min_max);
    }
  }
};

template<typename LOAD, typename STORE, typename T, typename ComputeType>
inline cudaError_t DispatchLayerNormMinMaxObserverBlockUncachedImpl(cudaStream_t stream, LOAD load,
                                                                    STORE store, const int64_t rows,
                                                                    const int64_t cols,
                                                                    const double epsilon,
                                                                    T* min_max) {
  return DispatchLayerNormMinMaxObserverBlockUncachedImplPackSize<LOAD, STORE, T, ComputeType>()(
      stream, load, store, rows, cols, epsilon, min_max);
}

template<typename LOAD, typename STORE, typename T, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNormMinMaxObserver(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                                const int64_t cols, const double epsilon, T* min_max) {
  if (cols <= 1024) {
    return DispatchLayerNormMinMaxObserverWarpImpl<LOAD, STORE, T, ComputeType>(
        stream, load, store, rows, cols, epsilon, min_max);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err =
          TryDispatchLayerNormMinMaxObserverBlockSMemImpl<LOAD, STORE, T, ComputeType>(
              stream, load, store, rows, cols, epsilon, min_max, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchLayerNormMinMaxObserverBlockUncachedImpl<LOAD, STORE, T, ComputeType>(
          stream, load, store, rows, cols, epsilon, min_max);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename T, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNormMinMaxObserver(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                                const int64_t cols, const double epsilon, T* min_max) {
  return DispatchLayerNormMinMaxObserverBlockUncachedImpl<LOAD, STORE, T, ComputeType>(
      stream, load, store, rows, cols, epsilon, min_max);
}

}  // namespace layer_norm

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_LAYER_NORM_MIN_MAX_OBSERVER_H_
