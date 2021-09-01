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

#ifndef ONEFLOW_CORE_CUDA_SOFTMAX_H_
#define ONEFLOW_CORE_CUDA_SOFTMAX_H_

#include <cub/cub.cuh>
#include <math_constants.h>
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace cuda {

namespace softmax {

constexpr int kWarpSize = 32;

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T a, const T b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T a, const T b) const { return max(a, b); }
};

template<template<typename> typename ReductionOp, typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<template<typename> typename ReductionOp, typename T, int block_size>
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
__inline__ __device__ T Inf();

template<>
__inline__ __device__ float Inf<float>() {
  return CUDART_INF_F;
}

template<>
__inline__ __device__ double Inf<double>() {
  return CUDART_INF;
}

template<typename T>
__inline__ __device__ T Exp(T x);

template<>
__inline__ __device__ float Exp<float>(float x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __expf(x);
#else
  return exp(x);
#endif
}

template<>
__inline__ __device__ double Exp<double>(double x) {
  return exp(x);
}

template<>
__inline__ __device__ half Exp<half>(half x) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __float2half(__expf(__half2float(x)));
#else
  return __float2half(exp(__half2float(x)));
#endif
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_SOFTMAX_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

inline int GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves) {
  int dev;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  int sm_count;
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  int tpm;
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  return std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
}

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

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
    const int64_t offset = row * row_size + col;
    pack.storage = *reinterpret_cast<const PackType<SRC, N>*>(src + offset);
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
    const int64_t offset = row * row_size + col;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *reinterpret_cast<PackType<DST, N>*>(dst + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

enum class SoftmaxType {
  kSoftmax = 0,
  kLogSoftmax = 1,
};

template<typename C, SoftmaxType T>
struct SoftmaxWarpImplOperation {
  __device__ void operator()(C* row_elem, const C warp_max_elem, C* thread_sum_elem) {
    *row_elem = Exp(*row_elem - warp_max_elem);
    *thread_sum_elem += *row_elem;
  }
  __device__ void operator()(C* row_elem, const C warp_sum_elem) {
    *row_elem = Div(*row_elem, warp_sum_elem);
  }
};

template<typename C>
struct SoftmaxWarpImplOperation<C, SoftmaxType::kLogSoftmax> {
  __device__ void operator()(C* row_elem, const C warp_max_elem, C* thread_sum_elem) {
    *row_elem -= warp_max_elem;
    *thread_sum_elem += Exp(*row_elem);
  }
  __device__ void operator()(C* row_elem, const C warp_sum_elem) {
    *row_elem -= SafeLog(warp_sum_elem);
  }
};

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding, SoftmaxType softmax_type>
__global__ void SoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;

  SoftmaxWarpImplOperation<ComputeType, softmax_type> f;

  for (int64_t row = global_thread_group_id * rows_per_access; row < rows;
       row += num_global_thread_group * rows_per_access) {
    ComputeType thread_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_max[row_id] = -Inf<ComputeType>();
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_id * pack_size, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            thread_max[row_id] = max(thread_max[row_id], row_buf[pack_id * pack_size + i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            row_buf[pack_id * pack_size + i] = -Inf<ComputeType>();
          }
        }
      }
    }
    ComputeType warp_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_max[row_id] = WarpAllReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
    }
    ComputeType thread_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_sum[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        f(&row_buf[i], warp_max[row_id], &thread_sum[row_id]);
      }
    }
    ComputeType warp_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
    }
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) { f(&row_buf[i], warp_sum[row_id]); }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
        }
      }
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, SoftmaxType softmax_type>
inline void DispatchSoftmaxWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int rows_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, rows_per_block);
  const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  const int grid_dim_x = GetNumBlocks(block_size, num_blocks, waves);

#define LaunchSoftmaxWarpImpl(padding)                                                      \
  SoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width, \
                  rows_per_access, padding, softmax_type>                                   \
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols);

  if (cols == cols_per_thread * thread_group_width) {
    LaunchSoftmaxWarpImpl(false)
  } else {
    LaunchSoftmaxWarpImpl(true)
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         SoftmaxType softmax_type>
typename std::enable_if<pack_size == 1, void>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(thread_group_width)                                                    \
  else if (cols <= (thread_group_width)*pack_size) {                                           \
    if (rows % 2 == 0) {                                                                       \
      DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,           \
                                     thread_group_width, 2, softmax_type>(stream, load, store, \
                                                                          rows, cols);         \
    } else {                                                                                   \
      DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,           \
                                     thread_group_width, 1, softmax_type>(stream, load, store, \
                                                                          rows, cols);         \
    }                                                                                          \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                               \
  else if (cols <= (col)*kWarpSize) {                                                      \
    DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, \
                                   softmax_type>(stream, load, store, rows, cols);         \
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
    UNIMPLEMENTED();
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         SoftmaxType softmax_type>
typename std::enable_if<pack_size == 2, void>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(thread_group_width)                                                    \
  else if (cols <= (thread_group_width)*pack_size) {                                           \
    if (rows % 2 == 0) {                                                                       \
      DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,           \
                                     thread_group_width, 2, softmax_type>(stream, load, store, \
                                                                          rows, cols);         \
    } else {                                                                                   \
      DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,           \
                                     thread_group_width, 1, softmax_type>(stream, load, store, \
                                                                          rows, cols);         \
    }                                                                                          \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                               \
  else if (cols <= (col)*kWarpSize) {                                                      \
    DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1, \
                                   softmax_type>(stream, load, store, rows, cols);         \
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
    UNIMPLEMENTED();
  }
}

template<typename LOAD, typename STORE, typename ComputeType, SoftmaxType softmax_type>
struct DispatchSoftmaxWarpImplPackSize {
  void operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols) {
    if (cols % 2 == 0) {
      DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2, softmax_type>(stream, load, store,
                                                                             rows, cols);
    } else {
      DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1, softmax_type>(stream, load, store,
                                                                             rows, cols);
    }
  }
};

template<typename C, SoftmaxType T>
struct SoftmaxBlockSMemImplOperation {
  __device__ void operator()(C* col_elem, const C row_max, C* thread_sum) {
    const C exp_x = Exp(*col_elem - row_max);
    *col_elem = exp_x;
    *thread_sum += exp_x;
  }
  __device__ void operator()(C* pack_elem, const C col_elem, const C row_sum) {
    *pack_elem = Div(col_elem, row_sum);
  }
};

template<typename C>
struct SoftmaxBlockSMemImplOperation<C, SoftmaxType::kLogSoftmax> {
  __device__ void operator()(C* col_elem, const C row_max, C* thread_sum) {
    const C x = *col_elem - row_max;
    *col_elem = x;
    *thread_sum += Exp(x);
  }
  __device__ void operator()(C* pack_elem, const C col_elem, const C row_sum) {
    *pack_elem = col_elem - SafeLog(row_sum);
  }
};

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         SoftmaxType softmax_type>
__global__ void SoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                     const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;

  SoftmaxBlockSMemImplOperation<ComputeType, softmax_type> f;

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -Inf<ComputeType>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        thread_max = max(thread_max, pack[i]);
      }
    }
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int col = tid; col < cols; col += block_size) { f(&buf[col], row_max, &thread_sum); }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { f(&pack[i], buf[i * num_packs + pack_id], row_sum); }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         SoftmaxType softmax_type>
inline bool TryDispatchSoftmaxBlockSMemImplBlockSize(cudaStream_t stream, LOAD load, STORE store,
                                                     const int64_t rows, const int64_t cols) {
  constexpr int waves = 32;

#define LaunchSoftmaxBlockSMemImpl(block_size_conf)                                        \
  const int grid_dim_x = GetNumBlocks(block_size_conf, rows, waves);                       \
  SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf, softmax_type> \
      <<<grid_dim_x, block_size_conf, smem, stream>>>(load, store, rows, cols);

  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType);
  int max_active_blocks_conf_1;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_1,
      SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1, softmax_type>,
      block_size_conf_1, smem));
  if (max_active_blocks_conf_1 <= 0) { return false; }
  int max_active_blocks_conf_4;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_4,
      SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4, softmax_type>,
      block_size_conf_4, smem));
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    LaunchSoftmaxBlockSMemImpl(block_size_conf_4) return true;
  }
  int max_active_blocks_conf_3;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_3,
      SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3, softmax_type>,
      block_size_conf_3, smem));
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    LaunchSoftmaxBlockSMemImpl(block_size_conf_3) return true;
  }
  int max_active_blocks_conf_2;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_2,
      SoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2, softmax_type>,
      block_size_conf_2, smem));
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    LaunchSoftmaxBlockSMemImpl(block_size_conf_2) return true;
  }
  LaunchSoftmaxBlockSMemImpl(block_size_conf_1) return true;
}

template<typename LOAD, typename STORE, typename ComputeType, SoftmaxType softmax_type>
struct TryDispatchSoftmaxBlockSMemImplPackSize {
  bool operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, softmax_type>(
          stream, load, store, rows, cols);
    } else {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, softmax_type>(
          stream, load, store, rows, cols);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, SoftmaxType softmax_type>
inline bool TryDispatchSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                            const int64_t rows, const int64_t cols) {
  return TryDispatchSoftmaxBlockSMemImplPackSize<LOAD, STORE, ComputeType, softmax_type>()(
      stream, load, store, rows, cols);
}

#define SoftmaxOperateReturnType \
  __device__ typename std::enable_if<T == SoftmaxType::kSoftmax, void>::type
#define LogSoftmaxOperateReturnType \
  __device__ typename std::enable_if<T == SoftmaxType::kLogSoftmax, void>::type

template<typename C, SoftmaxType T>
SoftmaxOperateReturnType SoftmaxBlockUncachedImplOperate(C* pack_elem, const C row_max,
                                                         const C row_sum) {
  *pack_elem = Div(Exp(*pack_elem - row_max), row_sum);
}

template<typename C, SoftmaxType T>
LogSoftmaxOperateReturnType SoftmaxBlockUncachedImplOperate(C* pack_elem, const C row_max,
                                                            const C row_sum) {
  *pack_elem = (*pack_elem - row_max) - SafeLog(row_sum);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size,
         SoftmaxType softmax_type>
__global__ void SoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                         const int64_t cols) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_max = -Inf<ComputeType>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_max = max(thread_max, pack[i]); }
    }
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_sum += Exp(pack[i] - row_max); }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        SoftmaxBlockUncachedImplOperate<ComputeType, softmax_type>(&pack[i], row_max, row_sum);
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, SoftmaxType softmax_type>
struct DispatchSoftmaxBlockUncachedImplPackSize {
  static constexpr int block_size = 1024;
  static constexpr int waves = 32;
  int grid_dim_x;
  void operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols) {
    grid_dim_x = GetNumBlocks(this->block_size, rows, this->waves);
#define LaunchSoftmaxBlockUncachedImpl(pack_size)                                         \
  SoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size, softmax_type> \
      <<<this->grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols);

    if (cols % 2 == 0) {
      LaunchSoftmaxBlockUncachedImpl(2)
    } else {
      LaunchSoftmaxBlockUncachedImpl(1)
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, SoftmaxType softmax_type>
inline void DispatchSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                             const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxBlockUncachedImplPackSize<LOAD, STORE, ComputeType, softmax_type>()(
      stream, load, store, rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType, SoftmaxType softmax_type>
inline void DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                            const int64_t cols) {
#define DispatchSoftmaxWarpImpl                                                                  \
  DispatchSoftmaxWarpImplPackSize<LOAD, STORE, ComputeType, softmax_type>()(stream, load, store, \
                                                                            rows, cols);
  if (cols <= 1024) {
    DispatchSoftmaxWarpImpl
  } else if (!TryDispatchSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, softmax_type>(
                 stream, load, store, rows, cols)) {
    DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, softmax_type>(stream, load, store,
                                                                             rows, cols);
  }
}

template<typename C, SoftmaxType T>
SoftmaxOperateReturnType SoftmaxGradDoSum(C* thread_sum, const C dy, const C y) {
  *thread_sum += dy * y;
}

template<typename C, SoftmaxType T>
LogSoftmaxOperateReturnType SoftmaxGradDoSum(C* thread_sum, const C dy, const C y) {
  (void)y;
  *thread_sum += dy;
}

template<typename C, SoftmaxType T>
SoftmaxOperateReturnType SoftmaxGradWarpImplOperate(C* dy, const C warp_sum, const C y) {
  *dy = (*dy - warp_sum) * y;
}

template<typename C, SoftmaxType T>
LogSoftmaxOperateReturnType SoftmaxGradWarpImplOperate(C* dy, const C warp_sum, const C y) {
  *dy = *dy - Exp(y) * warp_sum;
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, bool padding,
         SoftmaxType softmax_type>
__global__ void SoftmaxGradWarpImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
                                    const int64_t cols) {
  static_assert(cols_per_thread % pack_size == 0, "");
  constexpr int pack_per_thread = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  ComputeType y_buf[rows_per_access][cols_per_thread];
  ComputeType dy_buf[rows_per_access][cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows;
       row += num_global_thread_group * rows_per_access) {
    ComputeType thread_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_sum[row_id] = 0;
      ComputeType* row_y_buf = y_buf[row_id];
      ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          load_y.template load<pack_size>(row_y_buf + pack_id * pack_size, row + row_id, col);
          load_dy.template load<pack_size>(row_dy_buf + pack_id * pack_size, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            const int _index = pack_id * pack_size + i;
            SoftmaxGradDoSum<ComputeType, softmax_type>(&thread_sum[row_id], row_dy_buf[_index],
                                                        row_y_buf[_index]);
          }
        }
      }
    }
    ComputeType warp_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
    }
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_y_buf = y_buf[row_id];
      ComputeType* row_dy_buf = dy_buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          for (int i = 0; i < pack_size; ++i) {
            const int _index = pack_id * pack_size + i;
            SoftmaxGradWarpImplOperate<ComputeType, softmax_type>(
                &row_dy_buf[_index], warp_sum[row_id], row_y_buf[_index]);
          }
          store.template store<pack_size>(row_dy_buf + pack_id * pack_size, row + row_id, col);
        }
      }
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, bool padding,
         SoftmaxType softmax_type>
inline void LaunchSoftmaxGradWarpImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                      STORE store, const int64_t rows, const int64_t cols) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int rows_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, rows_per_block);
  const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  const int grid_dim_x = GetNumBlocks(block_size, num_blocks, waves);
  SoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread,
                      thread_group_width, rows_per_access, padding, softmax_type>
      <<<grid_dim_x, block_dim, 0, stream>>>(load_y, load_dy, store, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access, SoftmaxType softmax_type>
inline void DispatchSoftmaxGradWarpImplPadding(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                               STORE store, const int64_t rows,
                                               const int64_t cols) {
  if (cols == cols_per_thread * thread_group_width) {
    LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread,
                              thread_group_width, rows_per_access, false, softmax_type>(
        stream, load_y, load_dy, store, rows, cols);
  } else {
    LaunchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, cols_per_thread,
                              thread_group_width, rows_per_access, true, softmax_type>(
        stream, load_y, load_dy, store, rows, cols);
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         SoftmaxType softmax_type>
typename std::enable_if<pack_size == 1, void>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(thread_group_width)                                               \
  else if (cols <= (thread_group_width)*pack_size) {                                      \
    if (rows % 2 == 0) {                                                                  \
      DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,  \
                                         pack_size, thread_group_width, 2, softmax_type>( \
          stream, load_y, load_dy, store, rows, cols);                                    \
    } else {                                                                              \
      DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,  \
                                         pack_size, thread_group_width, 1, softmax_type>( \
          stream, load_y, load_dy, store, rows, cols);                                    \
    }                                                                                     \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                       \
  else if (cols <= (col)*kWarpSize) {                                                              \
    DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col,        \
                                       kWarpSize, 1, softmax_type>(stream, load_y, load_dy, store, \
                                                                   rows, cols);                    \
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
    UNIMPLEMENTED();
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         SoftmaxType softmax_type>
typename std::enable_if<pack_size == 2, void>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store, const int64_t rows,
    const int64_t cols) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(thread_group_width)                                               \
  else if (cols <= (thread_group_width)*pack_size) {                                      \
    if (rows % 2 == 0) {                                                                  \
      DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,  \
                                         pack_size, thread_group_width, 2, softmax_type>( \
          stream, load_y, load_dy, store, rows, cols);                                    \
    } else {                                                                              \
      DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,  \
                                         pack_size, thread_group_width, 2, softmax_type>( \
          stream, load_y, load_dy, store, rows, cols);                                    \
    }                                                                                     \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                       \
  else if (cols <= (col)*kWarpSize) {                                                              \
    DispatchSoftmaxGradWarpImplPadding<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, col,        \
                                       kWarpSize, 1, softmax_type>(stream, load_y, load_dy, store, \
                                                                   rows, cols);                    \
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
    UNIMPLEMENTED();
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         SoftmaxType softmax_type>
struct DispatchSoftmaxGradWarpImplPackSize {
  void operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                  const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
      DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, softmax_type>(
          stream, load_y, load_dy, store, rows, cols);
    } else {
      DispatchSoftmaxGradWarpImplCols<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, softmax_type>(
          stream, load_y, load_dy, store, rows, cols);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         SoftmaxType softmax_type>
inline void DispatchSoftmaxGradWarpImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                        STORE store, const int64_t rows, const int64_t cols) {
  DispatchSoftmaxGradWarpImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType, softmax_type>()(
      stream, load_y, load_dy, store, rows, cols);
}

template<typename C, SoftmaxType T>
SoftmaxOperateReturnType SoftmaxGradBlockSMemImplOperate(C* pack, const C dy, const C row_sum,
                                                         const C y) {
  *pack = (dy - row_sum) * y;
}

template<typename C, SoftmaxType T>
LogSoftmaxOperateReturnType SoftmaxGradBlockSMemImplOperate(C* pack, const C dy, const C row_sum,
                                                            const C y) {
  *pack = dy - Exp(y) * row_sum;
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, SoftmaxType softmax_type>
__global__ void SoftmaxGradBlockSMemImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                         const int64_t rows, const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
  auto* y_buf = reinterpret_cast<ComputeType*>(grad_shared_buf);
  auto* dy_buf = y_buf + cols;
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        y_buf[i * num_packs + pack_id] = y_pack[i];
        dy_buf[i * num_packs + pack_id] = dy_pack[i];
        SoftmaxGradDoSum<ComputeType, softmax_type>(&thread_sum, dy_pack[i], y_pack[i]);
      }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        const int _index = i * num_packs + pack_id;
        SoftmaxGradBlockSMemImplOperate<ComputeType, softmax_type>(&pack[i], dy_buf[_index],
                                                                   row_sum, y_buf[_index]);
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, SoftmaxType softmax_type>
inline void LaunchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                           STORE store, int smem, const int64_t rows,
                                           const int64_t cols) {
  constexpr int waves = 32;
  const int grid_dim_x = GetNumBlocks(block_size, rows, waves);
  SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size, softmax_type>
      <<<grid_dim_x, block_size, smem, stream>>>(load_y, load_dy, store, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         SoftmaxType softmax_type>
inline bool TryDispatchSoftmaxGradBlockSMemImplBlockSize(cudaStream_t stream, LOAD_Y load_y,
                                                         LOAD_DY load_dy, STORE store,
                                                         const int64_t rows, const int64_t cols) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType) * 2;
  int max_active_blocks_conf_1;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_1,
      SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_1,
                               softmax_type>,
      block_size_conf_1, smem));
  if (max_active_blocks_conf_1 <= 0) { return false; }
  int max_active_blocks_conf_4;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_4,
      SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_4,
                               softmax_type>,
      block_size_conf_4, smem));
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                   block_size_conf_4, softmax_type>(stream, load_y, load_dy, store,
                                                                    smem, rows, cols);
    return true;
  }
  int max_active_blocks_conf_3;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_3,
      SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_3,
                               softmax_type>,
      block_size_conf_3, smem));
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                   block_size_conf_3, softmax_type>(stream, load_y, load_dy, store,
                                                                    smem, rows, cols);
    return true;
  }
  int max_active_blocks_conf_2;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_2,
      SoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_2,
                               softmax_type>,
      block_size_conf_2, smem));
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size,
                                   block_size_conf_2, softmax_type>(stream, load_y, load_dy, store,
                                                                    smem, rows, cols);
    return true;
  }
  LaunchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf_1,
                                 softmax_type>(stream, load_y, load_dy, store, smem, rows, cols);
  return true;
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         SoftmaxType softmax_type>
struct TryDispatchSoftmaxGradBlockSMemImplPackSize {
  bool operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                  const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 2,
                                                          softmax_type>(stream, load_y, load_dy,
                                                                        store, rows, cols);
    } else {
      return TryDispatchSoftmaxGradBlockSMemImplBlockSize<LOAD_Y, LOAD_DY, STORE, ComputeType, 1,
                                                          softmax_type>(stream, load_y, load_dy,
                                                                        store, rows, cols);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         SoftmaxType softmax_type>
inline bool TryDispatchSoftmaxGradBlockSMemImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                                STORE store, const int64_t rows,
                                                const int64_t cols) {
  return TryDispatchSoftmaxGradBlockSMemImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                     softmax_type>()(stream, load_y, load_dy, store,
                                                                     rows, cols);
}

template<typename C, SoftmaxType T>
SoftmaxOperateReturnType SoftmaxGradBlockUncachedImplOperate(C* dy, const C row_sum, const C y) {
  *dy = (*dy - row_sum) * y;
}

template<typename C, SoftmaxType T>
LogSoftmaxOperateReturnType SoftmaxGradBlockUncachedImplOperate(C* dy, const C row_sum, const C y) {
  *dy -= Exp(y) * row_sum;
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size, SoftmaxType softmax_type>
__global__ void SoftmaxGradBlockUncachedImpl(LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                             const int64_t rows, const int64_t cols) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);

#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        SoftmaxGradDoSum<ComputeType, softmax_type>(&thread_sum, dy_pack[i], y_pack[i]);
      }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      load_y.template load<pack_size>(y_pack, row, pack_id * pack_size);
      load_dy.template load<pack_size>(dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        SoftmaxGradBlockUncachedImplOperate<ComputeType, softmax_type>(&dy_pack[i], row_sum,
                                                                       y_pack[i]);
      }
      store.template store<pack_size>(dy_pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         SoftmaxType softmax_type>
inline void LaunchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy,
                                               STORE store, const int64_t rows,
                                               const int64_t cols) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  const int grid_dim_x = GetNumBlocks(block_size, rows, waves);
  SoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, pack_size, block_size,
                               softmax_type>
      <<<grid_dim_x, block_size, 0, stream>>>(load_y, load_dy, store, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         SoftmaxType softmax_type>
struct DispatchSoftmaxGradBlockUncachedImplPackSize {
  void operator()(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                  const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0 && cols > kWarpSize) {
      LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 2, softmax_type>(
          stream, load_y, load_dy, store, rows, cols);
    } else {
      LaunchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, 1, softmax_type>(
          stream, load_y, load_dy, store, rows, cols);
    }
  }
};

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         SoftmaxType softmax_type>
inline void DispatchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, LOAD_Y load_y,
                                                 LOAD_DY load_dy, STORE store, const int64_t rows,
                                                 const int64_t cols) {
  return DispatchSoftmaxGradBlockUncachedImplPackSize<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                      softmax_type>()(stream, load_y, load_dy,
                                                                      store, rows, cols);
}

template<typename LOAD_Y, typename LOAD_DY, typename STORE, typename ComputeType,
         SoftmaxType softmax_type>
inline void DispatchSoftmaxGrad(cudaStream_t stream, LOAD_Y load_y, LOAD_DY load_dy, STORE store,
                                const int64_t rows, const int64_t cols) {
  if (cols <= 1024) {
    DispatchSoftmaxGradWarpImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, softmax_type>(
        stream, load_y, load_dy, store, rows, cols);
  } else if (!TryDispatchSoftmaxGradBlockSMemImpl<LOAD_Y, LOAD_DY, STORE, ComputeType,
                                                  softmax_type>(stream, load_y, load_dy, store,
                                                                rows, cols)) {
    DispatchSoftmaxGradBlockUncachedImpl<LOAD_Y, LOAD_DY, STORE, ComputeType, softmax_type>(
        stream, load_y, load_dy, store, rows, cols);
  }
}

}  // namespace softmax

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_SOFTMAX_H_
