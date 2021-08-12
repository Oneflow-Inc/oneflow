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

#ifndef ONEFLOW_CORE_CUDA_LOGSOFTMAX_H_
#define ONEFLOW_CORE_CUDA_LOGSOFTMAX_H_

#include <cub/cub.cuh>
#include <math_constants.h>
#include "softmax.cuh"

namespace oneflow {

namespace cuda {

namespace softmax {

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
__global__ void LogSoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                   STORE sub_result, STORE sum_result) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
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
      for (int i = 0; i < cols_per_thread; ++i) { row_buf[i] = row_buf[i] - warp_max[row_id]; }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          sub_result.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
        }
      }

#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = Exp(row_buf[i]);
        thread_sum[row_id] += row_buf[i];
      }
    }
    ComputeType warp_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
    }
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      sum_result.template store<1>(warp_sum + row_id, 0, row + row_id);
    }

#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) { row_buf[i] = Div(row_buf[i], warp_sum[row_id]); }
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
         int thread_group_width, int rows_per_access, bool padding>
inline void LaunchLogSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                     const int64_t rows, const int64_t cols, STORE sub_result,
                                     STORE sum_result) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int rows_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, rows_per_block);
  const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
  const int grid_dim_x = GetNumBlocks(block_size, num_blocks, waves);
  LogSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                     rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols, sub_result, sum_result);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access>
inline void DispatchLogSoftmaxWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                              const int64_t rows, const int64_t cols,
                                              STORE sub_result, STORE sum_result) {
  if (cols == cols_per_thread * thread_group_width) {
    LaunchLogSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                             thread_group_width, rows_per_access, false>(
        stream, load, store, rows, cols, sub_result, sum_result);
  } else {
    LaunchLogSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                             thread_group_width, rows_per_access, true>(
        stream, load, store, rows, cols, sub_result, sum_result);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, void>::type DispatchLogSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    STORE sub_result, STORE sum_result) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(thread_group_width)                                                     \
  else if (cols <= (thread_group_width)*pack_size) {                                            \
    if (rows % 2 == 0) {                                                                        \
      DispatchLogSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,         \
                                        thread_group_width, 2>(stream, load, store, rows, cols, \
                                                               sub_result, sum_result);         \
    } else {                                                                                    \
      DispatchLogSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,         \
                                        thread_group_width, 1>(stream, load, store, rows, cols, \
                                                               sub_result, sum_result);         \
    }                                                                                           \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                   \
  else if (cols <= (col)*kWarpSize) {                                                          \
    DispatchLogSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1>( \
        stream, load, store, rows, cols, sub_result, sum_result);                              \
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

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, void>::type DispatchLogSoftmaxWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    STORE sub_result, STORE sum_result) {
  if (cols <= 0) { UNIMPLEMENTED(); }
#define DEFINE_ONE_ELIF(thread_group_width)                                                     \
  else if (cols <= (thread_group_width)*pack_size) {                                            \
    if (rows % 2 == 0) {                                                                        \
      DispatchLogSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,         \
                                        thread_group_width, 2>(stream, load, store, rows, cols, \
                                                               sub_result, sum_result);         \
    } else {                                                                                    \
      DispatchLogSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size,         \
                                        thread_group_width, 1>(stream, load, store, rows, cols, \
                                                               sub_result, sum_result);         \
    }                                                                                           \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                   \
  else if (cols <= (col)*kWarpSize) {                                                          \
    DispatchLogSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size, col, kWarpSize, 1>( \
        stream, load, store, rows, cols, sub_result, sum_result);                              \
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

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLogSoftmaxWarpImplPackSize {
  void operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols, STORE sub_result, STORE sum_result) {
    if (cols % 2 == 0) {
      DispatchLogSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2>(stream, load, store, rows, cols,
                                                                  sub_result, sum_result);
    } else {
      DispatchLogSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1>(stream, load, store, rows, cols,
                                                                  sub_result, sum_result);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline void DispatchLogSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                       const int64_t rows, const int64_t cols, STORE sub_result,
                                       STORE sum_result) {
  DispatchLogSoftmaxWarpImplPackSize<LOAD, STORE, ComputeType>()(stream, load, store, rows, cols,
                                                                 sub_result, sum_result);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LogSoftmaxBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                        const int64_t cols, STORE sub_result, STORE sum_result) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
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
    for (int col = tid; col < cols; col += block_size) { buf[col] = buf[col] - row_max; }
    __syncthreads();

    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { pack[i] = buf[i * num_packs + pack_id]; }
      sub_result.template store<pack_size>(pack, row, pack_id * pack_size);
    }

    for (int col = tid; col < cols; col += block_size) {
      const ComputeType exp_x = Exp(buf[col]);
      buf[col] = exp_x;
      thread_sum += exp_x;
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    sum_result.template store<1>(&row_sum, 0, row);

    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = Div(buf[i * num_packs + pack_id], row_sum);
        thread_max = max(thread_max, pack[i]);
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
inline void LaunchLogSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store, int smem,
                                          const int64_t rows, const int64_t cols, STORE sub_result,
                                          STORE sum_result) {
  constexpr int waves = 32;
  const int grid_dim_x = GetNumBlocks(block_size, rows, waves);
  LogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols, sub_result, sum_result);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline bool TryDispatchLogSoftmaxBlockSMemImplBlockSize(cudaStream_t stream, LOAD load, STORE store,
                                                        const int64_t rows, const int64_t cols,
                                                        STORE sub_result, STORE sum_result) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType);
  int max_active_blocks_conf_1;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_1,
      LogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>,
      block_size_conf_1, smem));
  if (max_active_blocks_conf_1 <= 0) { return false; }
  int max_active_blocks_conf_4;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_4,
      LogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>,
      block_size_conf_4, smem));
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    LaunchLogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4>(
        stream, load, store, smem, rows, cols, sub_result, sum_result);
    return true;
  }
  int max_active_blocks_conf_3;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_3,
      LogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>,
      block_size_conf_3, smem));
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    LaunchLogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3>(
        stream, load, store, smem, rows, cols, sub_result, sum_result);
    return true;
  }
  int max_active_blocks_conf_2;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_2,
      LogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>,
      block_size_conf_2, smem));
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    LaunchLogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2>(
        stream, load, store, smem, rows, cols, sub_result, sum_result);
    return true;
  }
  LaunchLogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>(
      stream, load, store, smem, rows, cols, sub_result, sum_result);
  return true;
}

template<typename LOAD, typename STORE, typename ComputeType>
struct TryDispatchLogSoftmaxBlockSMemImplPackSize {
  bool operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols, STORE sub_result, STORE sum_result) {
    if (cols % 2 == 0) {
      return TryDispatchLogSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, sub_result, sum_result);
    } else {
      return TryDispatchLogSoftmaxBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, sub_result, sum_result);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline bool TryDispatchLogSoftmaxBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                               const int64_t rows, const int64_t cols,
                                               STORE sub_result, STORE sum_result) {
  return TryDispatchLogSoftmaxBlockSMemImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, sub_result, sum_result);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LogSoftmaxBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                            const int64_t cols, STORE sub_result,
                                            STORE sum_result) {
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
      for (int i = 0; i < pack_size; ++i) { pack[i] = pack[i] - row_max; }
      sub_result.template store<pack_size>(pack, row, pack_id * pack_size);

#pragma unroll
      for (int i = 0; i < pack_size; ++i) { thread_sum += Exp(pack[i]); }
    }
    const ComputeType row_sum = BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);

    sum_result.template store<1>(&row_sum, 0, row);

    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { pack[i] = Div(Exp(pack[i] - row_max), row_sum); }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline void LaunchLogSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                              const int64_t rows, const int64_t cols,
                                              STORE sub_result, STORE sum_result) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  const int grid_dim_x = GetNumBlocks(block_size, rows, waves);
  LogSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols, sub_result, sum_result);
}

template<typename LOAD, typename STORE, typename ComputeType>
struct DispatchLogSoftmaxBlockUncachedImplPackSize {
  void operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols, STORE sub_result, STORE sum_result) {
    if (cols % 2 == 0) {
      LaunchLogSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 2>(stream, load, store, rows,
                                                                     cols, sub_result, sum_result);
    } else {
      LaunchLogSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType, 1>(stream, load, store, rows,
                                                                     cols, sub_result, sum_result);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType>
inline void DispatchLogSoftmaxBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                const int64_t rows, const int64_t cols,
                                                STORE sub_result, STORE sum_result) {
  return DispatchLogSoftmaxBlockUncachedImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, sub_result, sum_result);
}

template<typename LOAD, typename STORE, typename ComputeType>
inline void DispatchLogSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                               const int64_t cols, STORE sub_result, STORE sum_result) {
  if (cols <= 1024) {
    DispatchLogSoftmaxWarpImpl<LOAD, STORE, ComputeType>(stream, load, store, rows, cols,
                                                         sub_result, sum_result);
  } else if (!TryDispatchLogSoftmaxBlockSMemImpl<LOAD, STORE, ComputeType>(
                 stream, load, store, rows, cols, sub_result, sum_result)) {
    DispatchLogSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType>(stream, load, store, rows, cols,
                                                                  sub_result, sum_result);
  }
}

}  // namespace softmax

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_LOGSOFTMAX_H_
