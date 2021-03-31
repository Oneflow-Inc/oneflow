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

namespace oneflow {

namespace cuda {

namespace softmax {

constexpr int kWarpSize = 32;

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> typename ReductionOp, typename T>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
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
struct GetComputeType {
  using type = T;
};

template<>
struct GetComputeType<half> {
  using type = float;
};

template<typename T, int N>
struct GetPackType;

template<typename T>
struct GetPackType<T, 1> {
  using type = T;
};

template<>
struct GetPackType<half, 2> {
  using type = half2;
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

template<typename SRC, typename DST, int N>
struct MultiFetch {
  __device__ void operator()(DST* dst, const SRC* src) const {
    Pack<SRC, N> pack;
    pack.storage = *reinterpret_cast<const PackType<SRC, N>*>(src);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
};

template<typename SRC>
struct UnaryMultiFetch {
  template<typename DST, int N>
  __device__ void fetch(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    int64_t offset = row * row_size + col;
    pack.storage = *reinterpret_cast<const PackType<SRC, N>*>(src + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }

  const SRC* src;
  int64_t row_size;
};

template<typename SRC>
struct GradMultiFetch {
  template<typename DST, int N>
  __device__ void fetch(DST* y_buf, DST* dy_buf, int64_t row, int64_t col) const {
    Pack<SRC, N> y_pack, dy_pack;
    int64_t offset = row * row_size + col;
    y_pack.storage = *reinterpret_cast<const PackType<SRC, N>*>(y + offset);
    dy_pack.storage = *reinterpret_cast<const PackType<SRC, N>*>(dy + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      y_buf[i] = static_cast<DST>(y_pack.elem[i]);
      dy_buf[i] = static_cast<DST>(dy_pack.elem[i]);
    }
  }

  const SRC* y;
  const SRC* dy;
  int64_t row_size;
};

template<typename SRC, typename DST, int N>
struct MultiStore {
  __device__ void operator()(DST* dst, const SRC* src) const {
    Pack<DST, N> pack;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *reinterpret_cast<PackType<DST, N>*>(dst) = pack.storage;
  }
};

template<typename FETCH, typename T, int pack_size, int cols_per_thread, bool padding>
__global__ void SoftmaxWarpImpl(FETCH fetch, const int64_t rows, const int64_t cols, const T* x,
                                T* y) {
  static_assert(cols_per_thread % pack_size == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * kWarpSize);
  using ComputeType = typename GetComputeType<T>::type;
  ComputeType buf[cols_per_thread];
  const int global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_warp = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  for (int64_t row = global_warp_id; row < rows; row += num_global_warp) {
    const int64_t row_offset = row * cols;
    const T* row_x = x + row_offset;
    T* row_y = y + row_offset;
    ComputeType thread_max = -Inf<ComputeType>();
#pragma unroll
    for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
      const int col = (pack_id * kWarpSize + lane_id) * pack_size;
      if (!padding || col < cols) {
        fetch.template fetch<ComputeType, pack_size>(buf + pack_id * pack_size, row, col);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          thread_max = max(thread_max, buf[pack_id * pack_size + i]);
        }
      } else {
#pragma unroll
        for (int i = 0; i < pack_size; ++i) { buf[pack_id * pack_size + i] = -Inf<ComputeType>(); }
      }
    }
    const ComputeType warp_max = WarpAllReduce<MaxOp, ComputeType>(thread_max);
    ComputeType thread_sum = 0;
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
      buf[i] = exp(buf[i] - warp_max);
      thread_sum += buf[i];
    }
    const ComputeType warp_sum = WarpAllReduce<SumOp, ComputeType>(thread_sum);
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) { buf[i] = buf[i] / warp_sum; }
#pragma unroll
    for (int i = 0; i < num_packs; ++i) {
      const int col = (i * kWarpSize + lane_id) * pack_size;
      if (!padding || col < cols) {
        MultiStore<ComputeType, T, pack_size>()(row_y + col, buf + i * pack_size);
      }
    }
  }
}

template<typename FETCH, typename T, int pack_size, int cols_per_thread, bool padding>
inline void LaunchSoftmaxWarpImpl(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                  const int64_t cols, const T* x, T* y) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % kWarpSize == 0, "");
  constexpr int rows_per_block = block_size / kWarpSize;
  dim3 block_dim(kWarpSize, rows_per_block);
  const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  const int grid_dim_x = GetNumBlocks(block_size, num_blocks, waves);
  SoftmaxWarpImpl<FETCH, T, pack_size, cols_per_thread, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(fetch, rows, cols, x, y);
}

template<typename FETCH, typename T, int pack_size, int cols_per_thread>
inline void DispatchSoftmaxWarpImplPadding(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                           const int64_t cols, const T* x, T* y) {
  if (cols == cols_per_thread * kWarpSize) {
    LaunchSoftmaxWarpImpl<FETCH, T, pack_size, cols_per_thread, false>(stream, fetch, rows, cols, x,
                                                                       y);
  } else {
    LaunchSoftmaxWarpImpl<FETCH, T, pack_size, cols_per_thread, true>(stream, fetch, rows, cols, x,
                                                                      y);
  }
}

template<typename FETCH, typename T, int pack_size>
typename std::enable_if<pack_size == 1, void>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols, const T* x, T* y) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(col)                                                                   \
  else if (cols <= (col)*kWarpSize) {                                                          \
    DispatchSoftmaxWarpImplPadding<FETCH, T, pack_size, col>(stream, fetch, rows, cols, x, y); \
  }
  DEFINE_ONE_ELIF(1)
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

template<typename FETCH, typename T, int pack_size>
typename std::enable_if<pack_size == 2, void>::type DispatchSoftmaxWarpImplCols(
    cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols, const T* x, T* y) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(col)                                                                   \
  else if (cols <= (col)*kWarpSize) {                                                          \
    DispatchSoftmaxWarpImplPadding<FETCH, T, pack_size, col>(stream, fetch, rows, cols, x, y); \
  }
  DEFINE_ONE_ELIF(2)
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

template<typename FETCH, typename T>
struct DispatchSoftmaxWarpImplPackSize {
  void operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const T* x, T* y) {
    DispatchSoftmaxWarpImplCols<FETCH, T, 1>(stream, fetch, rows, cols, x, y);
  }
};

template<typename FETCH>
struct DispatchSoftmaxWarpImplPackSize<FETCH, half> {
  void operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const half* x, half* y) {
    if (cols % 2 == 0 && cols > kWarpSize) {
      DispatchSoftmaxWarpImplCols<FETCH, half, 2>(stream, fetch, rows, cols, x, y);
    } else {
      DispatchSoftmaxWarpImplCols<FETCH, half, 1>(stream, fetch, rows, cols, x, y);
    }
  }
};

template<typename FETCH, typename T>
inline void DispatchSoftmaxWarpImpl(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                    const int64_t cols, const T* x, T* y) {
  DispatchSoftmaxWarpImplPackSize<FETCH, T>()(stream, fetch, rows, cols, x, y);
}

template<typename FETCH, typename T, int pack_size, int block_size>
__global__ void SoftmaxBlockSMemImpl(FETCH fetch, const int64_t rows, const int64_t cols,
                                     const T* x, T* y) {
  using ComputeType = typename GetComputeType<T>::type;
  extern __shared__ __align__(sizeof(ComputeType)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t row_offset = row * cols;
    const T* row_x = x + row_offset;
    T* row_y = y + row_offset;
    ComputeType thread_max = -Inf<ComputeType>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      fetch.template fetch<ComputeType, pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        thread_max = max(thread_max, pack[i]);
      }
    }
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int col = tid; col < cols; col += block_size) {
      const ComputeType exp_x = exp(buf[col] - row_max);
      buf[col] = exp_x;
      thread_sum += exp_x;
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = buf[i * num_packs + pack_id] / row_sum;
        thread_max = max(thread_max, pack[i]);
      }
      MultiStore<ComputeType, T, pack_size>()(row_y + pack_id * pack_size, pack);
    }
  }
}

template<typename FETCH, typename T, int pack_size, int block_size>
inline void LaunchSoftmaxBlockSMemImpl(cudaStream_t stream, FETCH fetch, int smem,
                                       const int64_t rows, const int64_t cols, const T* x, T* y) {
  constexpr int waves = 32;
  const int grid_dim_x = GetNumBlocks(block_size, rows, waves);
  SoftmaxBlockSMemImpl<FETCH, T, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(fetch, rows, cols, x, y);
}

template<typename FETCH, typename T, int pack_size>
inline bool TryDispatchSoftmaxBlockSMemImplBlockSize(cudaStream_t stream, FETCH fetch,
                                                     const int64_t rows, const int64_t cols,
                                                     const T* x, T* y) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  const size_t smem = cols * sizeof(typename GetComputeType<T>::type);
  int max_active_blocks_conf_1;
  int max_active_blocks_conf_2;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_1, SoftmaxBlockSMemImpl<FETCH, T, pack_size, block_size_conf_1>,
      block_size_conf_1, smem));
  if (max_active_blocks_conf_1 <= 0) { return false; }
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_2, SoftmaxBlockSMemImpl<FETCH, T, pack_size, block_size_conf_2>,
      block_size_conf_2, smem));
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    LaunchSoftmaxBlockSMemImpl<FETCH, T, pack_size, block_size_conf_2>(stream, fetch, smem, rows,
                                                                       cols, x, y);
  } else {
    LaunchSoftmaxBlockSMemImpl<FETCH, T, pack_size, block_size_conf_1>(stream, fetch, smem, rows,
                                                                       cols, x, y);
  }
  return true;
}

template<typename FETCH, typename T>
struct TryDispatchSoftmaxBlockSMemImplPackSize {
  bool operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const T* x, T* y) {
    return TryDispatchSoftmaxBlockSMemImplBlockSize<FETCH, T, 1>(stream, fetch, rows, cols, x, y);
  }
};

template<typename FETCH>
struct TryDispatchSoftmaxBlockSMemImplPackSize<FETCH, half> {
  bool operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const half* x, half* y) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<FETCH, half, 2>(stream, fetch, rows, cols, x,
                                                                      y);
    } else {
      return TryDispatchSoftmaxBlockSMemImplBlockSize<FETCH, half, 1>(stream, fetch, rows, cols, x,
                                                                      y);
    }
  }
};

template<typename FETCH, typename T>
inline bool TryDispatchSoftmaxBlockSMemImpl(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                            const int64_t cols, const T* x, T* y) {
  return TryDispatchSoftmaxBlockSMemImplPackSize<FETCH, T>()(stream, fetch, rows, cols, x, y);
}

template<typename FETCH, typename T, int pack_size, int block_size>
__global__ void SoftmaxBlockUncachedImpl(FETCH fetch, const int64_t rows, const int64_t cols,
                                         const T* x, T* y) {
  using ComputeType = typename GetComputeType<T>::type;
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t row_offset = row * cols;
    const T* row_x = x + row_offset;
    T* row_y = y + row_offset;
    ComputeType thread_max = -Inf<ComputeType>();
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      fetch.template fetch<ComputeType, pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_max = max(thread_max, pack[i]); }
    }
    const ComputeType row_max = BlockAllReduce<MaxOp, ComputeType, block_size>(thread_max);
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      fetch.template fetch<ComputeType, pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_sum += exp(pack[i] - row_max); }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      fetch.template fetch<ComputeType, pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { pack[i] = exp(pack[i] - row_max) / row_sum; }
      MultiStore<ComputeType, T, pack_size>()(row_y + pack_id * pack_size, pack);
    }
  }
}

template<typename FETCH, typename T, int pack_size>
inline void LaunchSoftmaxBlockUncachedImpl(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                           const int64_t cols, const T* x, T* y) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  const int grid_dim_x = GetNumBlocks(block_size, rows, waves);
  SoftmaxBlockUncachedImpl<FETCH, T, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(fetch, rows, cols, x, y);
}

template<typename FETCH, typename T>
struct DispatchSoftmaxBlockUncachedImplPackSize {
  void operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const T* x, T* y) {
    LaunchSoftmaxBlockUncachedImpl<FETCH, T, 1>(stream, fetch, rows, cols, x, y);
  }
};

template<typename FETCH>
struct DispatchSoftmaxBlockUncachedImplPackSize<FETCH, half> {
  void operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const half* x, half* y) {
    if (cols % 2 == 0) {
      LaunchSoftmaxBlockUncachedImpl<FETCH, half, 2>(stream, fetch, rows, cols, x, y);
    } else {
      LaunchSoftmaxBlockUncachedImpl<FETCH, half, 1>(stream, fetch, rows, cols, x, y);
    }
  }
};

template<typename FETCH, typename T>
inline void DispatchSoftmaxBlockUncachedImpl(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                             const int64_t cols, const T* x, T* y) {
  return DispatchSoftmaxBlockUncachedImplPackSize<FETCH, T>()(stream, fetch, rows, cols, x, y);
}

template<typename FETCH, typename T>
inline void DispatchSoftmax(cudaStream_t stream, FETCH fetch, const int64_t rows,
                            const int64_t cols, const T* x, T* y) {
  if (cols <= 1024) {
    DispatchSoftmaxWarpImpl<FETCH, T>(stream, fetch, rows, cols, x, y);
  } else if (!TryDispatchSoftmaxBlockSMemImpl(stream, fetch, rows, cols, x, y)) {
    DispatchSoftmaxBlockUncachedImpl<FETCH, T>(stream, fetch, rows, cols, x, y);
  }
}

template<typename FETCH, typename T, int pack_size, int cols_per_thread, bool padding>
__global__ void SoftmaxGradWarpImpl(FETCH fetch, const int64_t rows, const int64_t cols, const T* y,
                                    const T* dy, T* dx) {
  static_assert(cols_per_thread % pack_size == 0, "");
  constexpr int pack_per_thread = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * kWarpSize);
  using ComputeType = typename GetComputeType<T>::type;
  ComputeType y_buf[cols_per_thread];
  ComputeType dy_buf[cols_per_thread];
  const int global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_warp = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  for (int64_t row = global_warp_id; row < rows; row += num_global_warp) {
    const int64_t row_offset = row * cols;
    const T* row_y = y + row_offset;
    const T* row_dy = dy + row_offset;
    T* row_dx = dx + row_offset;
    ComputeType thread_sum = 0;
#pragma unroll
    for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
      const int col = (pack_id * kWarpSize + lane_id) * pack_size;
      if (!padding || col < cols) {
        fetch.template fetch<ComputeType, pack_size>(y_buf + pack_id * pack_size,
                                                     dy_buf + pack_id * pack_size, row, col);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          thread_sum += y_buf[pack_id * pack_size + i] * dy_buf[pack_id * pack_size + i];
        }
      }
    }
    const ComputeType warp_sum = WarpAllReduce<SumOp, ComputeType>(thread_sum);
#pragma unroll
    for (int pack_id = 0; pack_id < pack_per_thread; ++pack_id) {
      const int col = (pack_id * kWarpSize + lane_id) * pack_size;
      if (!padding || col < cols) {
        for (int i = 0; i < pack_size; ++i) {
          dy_buf[pack_id * pack_size + i] =
              (dy_buf[pack_id * pack_size + i] - warp_sum) * y_buf[pack_id * pack_size + i];
        }
        MultiStore<ComputeType, T, pack_size>()(row_dx + col, dy_buf + pack_id * pack_size);
      }
    }
  }
}

template<typename FETCH, typename T, int pack_size, int cols_per_thread, bool padding>
inline void LaunchSoftmaxGradWarpImpl(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                      const int64_t cols, const T* y, const T* dy, T* dx) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % kWarpSize == 0, "");
  constexpr int rows_per_block = block_size / kWarpSize;
  dim3 block_dim(kWarpSize, rows_per_block);
  const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  const int grid_dim_x = GetNumBlocks(block_size, num_blocks, waves);
  SoftmaxGradWarpImpl<FETCH, T, pack_size, cols_per_thread, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(fetch, rows, cols, y, dy, dx);
}

template<typename FETCH, typename T, int pack_size, int cols_per_thread>
inline void DispatchSoftmaxGradWarpImplPadding(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                               const int64_t cols, const T* y, const T* dy, T* dx) {
  if (cols == cols_per_thread * kWarpSize) {
    LaunchSoftmaxGradWarpImpl<FETCH, T, pack_size, cols_per_thread, false>(stream, fetch, rows,
                                                                           cols, y, dy, dx);
  } else {
    LaunchSoftmaxGradWarpImpl<FETCH, T, pack_size, cols_per_thread, true>(stream, fetch, rows, cols,
                                                                          y, dy, dx);
  }
}

template<typename FETCH, typename T, int pack_size>
typename std::enable_if<pack_size == 1, void>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols, const T* y,
    const T* dy, T* dx) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(col)                                                                       \
  else if (cols <= (col)*kWarpSize) {                                                              \
    DispatchSoftmaxGradWarpImplPadding<FETCH, T, pack_size, col>(stream, fetch, rows, cols, y, dy, \
                                                                 dx);                              \
  }
  DEFINE_ONE_ELIF(1)
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

template<typename FETCH, typename T, int pack_size>
typename std::enable_if<pack_size == 2, void>::type DispatchSoftmaxGradWarpImplCols(
    cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols, const T* y,
    const T* dy, T* dx) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(col)                                                                       \
  else if (cols <= (col)*kWarpSize) {                                                              \
    DispatchSoftmaxGradWarpImplPadding<FETCH, T, pack_size, col>(stream, fetch, rows, cols, y, dy, \
                                                                 dx);                              \
  }
  DEFINE_ONE_ELIF(2)
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

template<typename FETCH, typename T>
struct DispatchSoftmaxGradWarpImplPackSize {
  void operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const T* y, const T* dy, T* dx) {
    DispatchSoftmaxGradWarpImplCols<FETCH, T, 1>(stream, fetch, rows, cols, y, dy, dx);
  }
};

template<typename FETCH>
struct DispatchSoftmaxGradWarpImplPackSize<FETCH, half> {
  void operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const half* y, const half* dy, half* dx) {
    if (cols % 2 == 0 && cols > kWarpSize) {
      DispatchSoftmaxGradWarpImplCols<FETCH, half, 2>(stream, fetch, rows, cols, y, dy, dx);
    } else {
      DispatchSoftmaxGradWarpImplCols<FETCH, half, 1>(stream, fetch, rows, cols, y, dy, dx);
    }
  }
};

template<typename FETCH, typename T>
inline void DispatchSoftmaxGradWarpImpl(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                        const int64_t cols, const T* y, const T* dy, T* dx) {
  DispatchSoftmaxGradWarpImplPackSize<FETCH, T>()(stream, fetch, rows, cols, y, dy, dx);
}

template<typename FETCH, typename T, int pack_size, int block_size>
__global__ void SoftmaxGradBlockSMemImpl(FETCH fetch, const int64_t rows, const int64_t cols,
                                         const T* y, const T* dy, T* dx) {
  using ComputeType = typename GetComputeType<T>::type;
  extern __shared__ __align__(sizeof(ComputeType)) unsigned char grad_shared_buf[];
  auto* y_buf = reinterpret_cast<ComputeType*>(grad_shared_buf);
  auto* dy_buf = y_buf + cols;
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t row_offset = row * cols;
    const T* row_y = y + row_offset;
    const T* row_dy = dy + row_offset;
    T* row_dx = dx + row_offset;
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      fetch.template fetch<ComputeType, pack_size>(y_pack, dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        y_buf[i * num_packs + pack_id] = y_pack[i];
        dy_buf[i * num_packs + pack_id] = dy_pack[i];
        thread_sum += y_pack[i] * dy_pack[i];
      }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = (dy_buf[i * num_packs + pack_id] - row_sum) * y_buf[i * num_packs + pack_id];
      }
      MultiStore<ComputeType, T, pack_size>()(row_dx + pack_id * pack_size, pack);
    }
  }
}

template<typename FETCH, typename T, int pack_size, int block_size>
inline void LaunchSoftmaxGradBlockSMemImpl(cudaStream_t stream, FETCH fetch, int smem,
                                           const int64_t rows, const int64_t cols, const T* y,
                                           const T* dy, T* dx) {
  constexpr int waves = 32;
  const int grid_dim_x = GetNumBlocks(block_size, rows, waves);
  SoftmaxGradBlockSMemImpl<FETCH, T, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(fetch, rows, cols, y, dy, dx);
}

template<typename FETCH, typename T, int pack_size>
inline bool TryDispatchSoftmaxGradBlockSMemImplBlockSize(cudaStream_t stream, FETCH fetch,
                                                         const int64_t rows, const int64_t cols,
                                                         const T* y, const T* dy, T* dx) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  const size_t smem = cols * sizeof(typename GetComputeType<T>::type) * 2;
  int max_active_blocks_conf_1;
  int max_active_blocks_conf_2;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_1, SoftmaxGradBlockSMemImpl<FETCH, T, pack_size, block_size_conf_1>,
      block_size_conf_1, smem));
  if (max_active_blocks_conf_1 <= 0) { return false; }
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_2, SoftmaxGradBlockSMemImpl<FETCH, T, pack_size, block_size_conf_2>,
      block_size_conf_2, smem));
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    LaunchSoftmaxGradBlockSMemImpl<FETCH, T, pack_size, block_size_conf_2>(stream, fetch, smem,
                                                                           rows, cols, y, dy, dx);
  } else {
    LaunchSoftmaxGradBlockSMemImpl<FETCH, T, pack_size, block_size_conf_1>(stream, fetch, smem,
                                                                           rows, cols, y, dy, dx);
  }
  return true;
}

template<typename FETCH, typename T>
struct TryDispatchSoftmaxGradBlockSMemImplPackSize {
  bool operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const T* y, const T* dy, T* dx) {
    return TryDispatchSoftmaxGradBlockSMemImplBlockSize<FETCH, T, 1>(stream, fetch, rows, cols, y,
                                                                     dy, dx);
  }
};

template<typename FETCH>
struct TryDispatchSoftmaxGradBlockSMemImplPackSize<FETCH, half> {
  bool operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const half* y, const half* dy, half* dx) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxGradBlockSMemImplBlockSize<FETCH, half, 2>(stream, fetch, rows, cols,
                                                                          y, dy, dx);
    } else {
      return TryDispatchSoftmaxGradBlockSMemImplBlockSize<FETCH, half, 1>(stream, fetch, rows, cols,
                                                                          y, dy, dx);
    }
  }
};

template<typename FETCH, typename T>
inline bool TryDispatchSoftmaxGradBlockSMemImpl(cudaStream_t stream, FETCH fetch,
                                                const int64_t rows, const int64_t cols, const T* y,
                                                const T* dy, T* dx) {
  return TryDispatchSoftmaxGradBlockSMemImplPackSize<FETCH, T>()(stream, fetch, rows, cols, y, dy,
                                                                 dx);
}

template<typename FETCH, typename T, int pack_size, int block_size>
__global__ void SoftmaxGradBlockUncachedImpl(FETCH fetch, const int64_t rows, const int64_t cols,
                                             const T* y, const T* dy, T* dx) {
  using ComputeType = typename GetComputeType<T>::type;
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t row_offset = row * cols;
    const T* row_y = y + row_offset;
    const T* row_dy = dy + row_offset;
    T* row_dx = dx + row_offset;
    ComputeType thread_sum = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      fetch.template fetch<ComputeType, pack_size>(y_pack, dy_pack, row, pack_id * pack_size);

#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_sum += y_pack[i] * dy_pack[i]; }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType y_pack[pack_size];
      ComputeType dy_pack[pack_size];
      fetch.template fetch<ComputeType, pack_size>(y_pack, dy_pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { dy_pack[i] = (dy_pack[i] - row_sum) * y_pack[i]; }
      MultiStore<ComputeType, T, pack_size>()(row_dx + pack_id * pack_size, dy_pack);
    }
  }
}

template<typename FETCH, typename T, int pack_size>
inline void LaunchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                               const int64_t cols, const T* y, const T* dy, T* dx) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  const int grid_dim_x = GetNumBlocks(block_size, rows, waves);
  SoftmaxGradBlockUncachedImpl<FETCH, T, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(fetch, rows, cols, y, dy, dx);
}

template<typename FETCH, typename T>
struct DispatchSoftmaxGradBlockUncachedImplPackSize {
  void operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const T* y, const T* dy, T* dx) {
    LaunchSoftmaxGradBlockUncachedImpl<FETCH, T, 1>(stream, fetch, rows, cols, y, dy, dx);
  }
};

template<typename FETCH>
struct DispatchSoftmaxGradBlockUncachedImplPackSize<FETCH, half> {
  void operator()(cudaStream_t stream, FETCH fetch, const int64_t rows, const int64_t cols,
                  const half* y, const half* dy, half* dx) {
    if (cols % 2 == 0 && cols > kWarpSize) {
      LaunchSoftmaxGradBlockUncachedImpl<FETCH, half, 2>(stream, fetch, rows, cols, y, dy, dx);
    } else {
      LaunchSoftmaxGradBlockUncachedImpl<FETCH, half, 1>(stream, fetch, rows, cols, y, dy, dx);
    }
  }
};

template<typename FETCH, typename T>
inline void DispatchSoftmaxGradBlockUncachedImpl(cudaStream_t stream, FETCH fetch,
                                                 const int64_t rows, const int64_t cols, const T* y,
                                                 const T* dy, T* dx) {
  return DispatchSoftmaxGradBlockUncachedImplPackSize<FETCH, T>()(stream, fetch, rows, cols, y, dy,
                                                                  dx);
}

template<typename FETCH, typename T>
inline void DispatchSoftmaxGrad(cudaStream_t stream, FETCH fetch, const int64_t rows,
                                const int64_t cols, const T* y, const T* dy, T* dx) {
  if (cols <= 1024) {
    DispatchSoftmaxGradWarpImpl<FETCH, T>(stream, fetch, rows, cols, y, dy, dx);
  } else if (!TryDispatchSoftmaxGradBlockSMemImpl<FETCH, T>(stream, fetch, rows, cols, y, dy, dx)) {
    DispatchSoftmaxGradBlockUncachedImpl<FETCH, T>(stream, fetch, rows, cols, y, dy, dx);
  }
}

}  // namespace softmax

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_SOFTMAX_H_
