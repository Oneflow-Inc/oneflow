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

#ifndef ONEFLOW_CORE_CUDA_RMS_NORM_H_
#define ONEFLOW_CORE_CUDA_RMS_NORM_H_

#include "oneflow/core/cuda/layer_norm.cuh"

namespace oneflow {
namespace cuda {
namespace rms_norm {

constexpr int kWarpSize = 32;

template<typename T>
__inline__ __device__ T WarpReduceSum(T val) {
  for (int mask = 16; mask > 0; mask /= 2) { val += __shfl_down_sync(0xffffffff, val, mask); }
  return val;
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access, bool padding>
__global__ void RmsNormWarpImpl(LOAD load, STORE store, const int nrow, const int ncol,
                                const double eps, ComputeType* inv_rms) {
  static_assert(max_cols_per_thread % pack_size == 0, "");
  static_assert(min_cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int max_packs = max_cols_per_thread / pack_size;
  constexpr int min_packs = min_cols_per_thread / pack_size;
  assert(ncol <= max_cols_per_thread * thread_group_width);

  ComputeType buf[rows_per_access][max_cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_groups = gridDim.x * blockDim.y;
  for (int row_i = global_thread_group_id; row_i < nrow; row_i += num_global_thread_groups) {
    ComputeType thread_square_sum[rows_per_access];
#pragma unroll
    for (int row_j = 0; row_j < rows_per_access; ++row_j) {
      thread_square_sum[row_j] = 0;
      ComputeType* row_buf = buf[row_j];
      const int row = row_i * rows_per_access + row_j;
#pragma unroll
      for (int pack_i = 0; pack_i < min_packs; ++pack_i) {
        const int pack_offset = pack_i * pack_size;
        const int col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        load.template load<pack_size>(row_buf + pack_offset, row, col);
#pragma unroll
        for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
          thread_square_sum[row_j] += row_buf[pack_offset + pack_j] * row_buf[pack_offset + pack_j];
        }
      }
#pragma unroll
      for (int pack_i = min_packs; pack_i < max_packs; ++pack_i) {
        const int pack_offset = pack_i * pack_size;
        const int col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        if (!padding || col < ncol) {
          load.template load<pack_size>(row_buf + pack_offset, row, col);
#pragma unroll
          for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
            thread_square_sum[row_j] +=
                row_buf[pack_offset + pack_j] * row_buf[pack_offset + pack_j];
          }
        } else {
#pragma unroll
          for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
            row_buf[pack_i * pack_size + pack_j] = 0;
          }
        }
      }
    }
    ComputeType warp_square_sum[rows_per_access];
#pragma unroll
    for (int row_j = 0; row_j < rows_per_access; ++row_j) {
      const int row = row_i * rows_per_access + row_j;
      ComputeType* row_buf = buf[row_j];
      warp_square_sum[row_j] =
          layer_norm::WarpAllReduce<layer_norm::SumOp, ComputeType, thread_group_width>(
              thread_square_sum[row_j]);
      ComputeType row_square_mean =
          layer_norm::Div(warp_square_sum[row_j], static_cast<ComputeType>(ncol));
      ComputeType row_inv_rms = layer_norm::Rsqrt(row_square_mean + static_cast<ComputeType>(eps));
      if (threadIdx.x == 0) { inv_rms[row] = row_inv_rms; }
#pragma unroll
      for (int col = 0; col < max_cols_per_thread; ++col) { row_buf[col] *= row_inv_rms; }
#pragma unroll
      for (int pack_i = 0; pack_i < min_packs; ++pack_i) {
        const int col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        store.template store<pack_size>(row_buf + pack_i * pack_size, row, col);
      }
#pragma unroll
      for (int pack_i = min_packs; pack_i < max_packs; ++pack_i) {
        const int col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        if (!padding || col < ncol) {
          store.template store<pack_size>(row_buf + pack_i * pack_size, row, col);
        }
      }
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access, bool padding>
cudaError_t LaunchRmsNormWarpImpl(cudaStream_t stream, LOAD load, STORE store, const int64_t nrow,
                                  const int64_t ncol, const double eps, ComputeType* inv_rms) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  const int64_t num_blocks =
      (nrow / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err = layer_norm::GetNumBlocks(
        RmsNormWarpImpl<LOAD, STORE, ComputeType, pack_size, max_cols_per_thread,
                        min_cols_per_thread, thread_group_width, rows_per_access, padding>,
        block_size, 0, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  RmsNormWarpImpl<LOAD, STORE, ComputeType, pack_size, max_cols_per_thread, min_cols_per_thread,
                  thread_group_width, rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, static_cast<int>(nrow),
                                             static_cast<int>(ncol), eps, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access>
cudaError_t DispatchLaunchRmsNormWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                 const int64_t nrow, const int64_t ncol,
                                                 const double eps, ComputeType* inv_rms) {
  if (ncol == max_cols_per_thread * thread_group_width) {
    // when not padding, min_cols_per_thread must equals to max_cols_per_thread, pass
    // max_cols_per_thread as min_cols_per_thread and max_cols_per_thread param.
    return LaunchRmsNormWarpImpl<LOAD, STORE, ComputeType, pack_size, max_cols_per_thread,
                                 max_cols_per_thread, thread_group_width, rows_per_access, false>(
        stream, load, store, nrow, ncol, eps, inv_rms);
  } else {
    return LaunchRmsNormWarpImpl<LOAD, STORE, ComputeType, pack_size, max_cols_per_thread,
                                 min_cols_per_thread, thread_group_width, rows_per_access, true>(
        stream, load, store, nrow, ncol, eps, inv_rms);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLaunchRmsNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t nrow, const int64_t ncol,
    const double eps, ComputeType* inv_rms) {
  if (ncol <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                       \
  else if (ncol <= (thread_group_width)*pack_size) {                                              \
    if (nrow % 2 == 0) {                                                                          \
      return DispatchLaunchRmsNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                                  0, thread_group_width, 2>(                      \
          stream, load, store, nrow, ncol, eps, inv_rms);                                         \
    } else {                                                                                      \
      return DispatchLaunchRmsNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                                  0, thread_group_width, 1>(                      \
          stream, load, store, nrow, ncol, eps, inv_rms);                                         \
    }                                                                                             \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col)                                                         \
  else if (ncol <= (max_col)*kWarpSize) {                                                         \
    return DispatchLaunchRmsNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, max_col,     \
                                                min_col, kWarpSize, 1>(stream, load, store, nrow, \
                                                                       ncol, eps, inv_rms);       \
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

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchLaunchRmsNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t nrow, const int64_t ncol,
    const double eps, ComputeType* inv_rms) {
  if (ncol <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                       \
  else if (ncol <= (thread_group_width)*pack_size) {                                              \
    if (nrow % 2 == 0) {                                                                          \
      return DispatchLaunchRmsNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                                  0, thread_group_width, 2>(                      \
          stream, load, store, nrow, ncol, eps, inv_rms);                                         \
    } else {                                                                                      \
      return DispatchLaunchRmsNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, pack_size, \
                                                  0, thread_group_width, 1>(                      \
          stream, load, store, nrow, ncol, eps, inv_rms);                                         \
    }                                                                                             \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col)                                                         \
  else if ((ncol <= (max_col)*kWarpSize) && (ncol > (min_col)*kWarpSize)) {                       \
    return DispatchLaunchRmsNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, max_col,     \
                                                min_col, kWarpSize, 1>(stream, load, store, nrow, \
                                                                       ncol, eps, inv_rms);       \
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

template<typename LOAD, typename STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormWarpImplPackSize(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t nrow, const int64_t ncol,
                                                  const double eps, ComputeType* inv_rms) {
  if (ncol % 2 == 0 && layer_norm::CanPackAs<LOAD>(load, 2)
      && layer_norm::CanPackAs<STORE>(store, 2)) {
    return DispatchLaunchRmsNormWarpImplCols<LOAD, STORE, ComputeType, 2>(stream, load, store, nrow,
                                                                          ncol, eps, inv_rms);
  } else {
    return DispatchLaunchRmsNormWarpImplCols<LOAD, STORE, ComputeType, 1>(stream, load, store, nrow,
                                                                          ncol, eps, inv_rms);
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                          const int64_t nrow, const int64_t ncol, const double eps,
                                          ComputeType* inv_rms) {
  return DispatchLaunchRmsNormWarpImplPackSize(stream, load, store, nrow, ncol, eps, inv_rms);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void RmsNormBlockSMemImpl(LOAD load, STORE store, const int nrow, const int ncol,
                                     const double eps, ComputeType* inv_rms) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  assert(ncol % pack_size == 0);
  const int num_packs = ncol / pack_size;
  for (int row = blockIdx.x; row < nrow; row += gridDim.x) {
    ComputeType thread_square_sum = 0;
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += block_size) {
      ComputeType pack[pack_size];
      const int col = pack_i * pack_size;
      load.template load<pack_size>(pack, row, col);
#pragma unroll
      for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
        buf[pack_i * pack_size + pack_j] = pack[pack_j];
        thread_square_sum += pack[pack_j] * pack[pack_j];
      }
    }
    ComputeType row_square_sum =
        layer_norm::BlockAllReduce<layer_norm::SumOp, ComputeType, block_size>(thread_square_sum);
    ComputeType row_square_mean = layer_norm::Div(row_square_sum, static_cast<ComputeType>(ncol));
    ComputeType row_inv_rms = layer_norm::Rsqrt(row_square_mean + static_cast<ComputeType>(eps));
    if (threadIdx.x == 0) { inv_rms[row] = row_inv_rms; }
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
        pack[pack_j] = buf[pack_i * pack_size + pack_j] * row_inv_rms;
      }
      const int col = pack_i * pack_size;
      store.template store<pack_size>(pack, row, col);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
cudaError_t LaunchRmsNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                       size_t smem_size, const int64_t nrow, const int64_t ncol,
                                       const double eps, ComputeType* inv_rms) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = layer_norm::GetNumBlocks(
        RmsNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>, block_size,
        smem_size, nrow, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  RmsNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem_size, stream>>>(load, store, nrow, ncol, eps, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
cudaError_t TryDispatchLaunchRmsNormBlockSMemImplBlockSize(cudaStream_t stream, LOAD load,
                                                           STORE store, const int64_t nrow,
                                                           const int64_t ncol, const double eps,
                                                           ComputeType* inv_rms, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem_size = ncol * sizeof(ComputeType);
  int max_active_blocks = 0;
  int num_blocks = 0;

#define SELECT_BLOCK_SIZE_CONF(block_size_conf)                                                  \
  {                                                                                              \
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(                             \
        &num_blocks, RmsNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf>, \
        block_size_conf, smem_size);                                                             \
    if (err != cudaSuccess) { return err; }                                                      \
    if (max_active_blocks == 0) {                                                                \
      if (num_blocks <= max_active_blocks) {                                                     \
        *success = false;                                                                        \
        return cudaSuccess;                                                                      \
      }                                                                                          \
      max_active_blocks = num_blocks;                                                            \
    } else {                                                                                     \
      if (num_blocks == max_active_blocks) {                                                     \
        *success = true;                                                                         \
        return LaunchRmsNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf>( \
            stream, load, store, smem_size, nrow, ncol, eps, inv_rms);                           \
      }                                                                                          \
    }                                                                                            \
  }

  SELECT_BLOCK_SIZE_CONF(block_size_conf_1)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_4)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_3)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_2)
#undef SELECT_BLOCK_SIZE_CONF

  *success = true;
  return LaunchRmsNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1>(
      stream, load, store, smem_size, nrow, ncol, eps, inv_rms);
}

template<typename LOAD, typename STORE, typename ComputeType>
cudaError_t TryDispatchLaunchRmsNormBlockSMemImplPackSize(cudaStream_t stream, LOAD load,
                                                          STORE store, const int64_t nrow,
                                                          const int64_t ncol, const double eps,
                                                          ComputeType* inv_rms, bool* success) {
  if (ncol % 4 == 0 && layer_norm::CanPackAs<LOAD>(load, 4)
      && layer_norm::CanPackAs<STORE>(store, 4)) {
    return TryDispatchLaunchRmsNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 4>(
        stream, load, store, nrow, ncol, eps, inv_rms, success);
  } else if (ncol % 2 == 0 && layer_norm::CanPackAs<LOAD>(load, 2)
             && layer_norm::CanPackAs<STORE>(store, 2)) {
    return TryDispatchLaunchRmsNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2>(
        stream, load, store, nrow, ncol, eps, inv_rms, success);
  } else {
    return TryDispatchLaunchRmsNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1>(
        stream, load, store, nrow, ncol, eps, inv_rms, success);
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
cudaError_t TryDispatchLaunchRmsNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t nrow, const int64_t ncol,
                                                  const double eps, ComputeType* inv_rms,
                                                  bool* success) {
  return TryDispatchLaunchRmsNormBlockSMemImplPackSize(stream, load, store, nrow, ncol, eps,
                                                       inv_rms, success);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void RmsNormBlockUncachedImpl(LOAD load, STORE store, const int nrow, const int ncol,
                                         const double eps, ComputeType* inv_rms) {
  assert(ncol % pack_size == 0);
  const int num_packs = ncol / pack_size;
  for (int row = blockIdx.x; row < nrow; row += gridDim.x) {
    ComputeType thread_square_sum = 0;
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += block_size) {
      ComputeType pack[pack_size];
      const int col = pack_i * pack_size;
      load.template load<pack_size>(pack, row, col);
#pragma unroll
      for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
        thread_square_sum += pack[pack_j] * pack[pack_j];
      }
    }
    ComputeType row_square_sum =
        layer_norm::BlockAllReduce<layer_norm::SumOp, ComputeType, block_size>(thread_square_sum);
    ComputeType row_square_mean = layer_norm::Div(row_square_sum, static_cast<ComputeType>(ncol));
    ComputeType row_inv_rms = layer_norm::Rsqrt(row_square_mean + static_cast<ComputeType>(eps));
    if (threadIdx.x == 0) { inv_rms[row] = row_inv_rms; }
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += block_size) {
      ComputeType pack[pack_size];
      const int col = pack_i * pack_size;
      load.template load<pack_size>(pack, row, col);
#pragma unroll
      for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
        pack[pack_j] = pack[pack_j] * row_inv_rms;
      }
      store.template store<pack_size>(pack, row, col);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size>
cudaError_t LaunchRmsNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t nrow, const int64_t ncol, const double eps,
                                           ComputeType* inv_rms) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = layer_norm::GetNumBlocks(
        RmsNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>, block_size, 0,
        nrow, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  RmsNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, nrow, ncol, eps, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormBlockUncachedImplPackSize(cudaStream_t stream, LOAD load,
                                                           STORE store, const int64_t nrow,
                                                           const int64_t ncol, const double eps,
                                                           ComputeType* inv_rms) {
  if (ncol % 4 == 0 && layer_norm::CanPackAs<LOAD>(load, 4)
      && layer_norm::CanPackAs<STORE>(store, 4)) {
    return LaunchRmsNormBlockUncachedImpl<LOAD, STORE, ComputeType, 4>(stream, load, store, nrow,
                                                                       ncol, eps, inv_rms);
  } else if (ncol % 2 == 0 && layer_norm::CanPackAs<LOAD>(load, 2)
             && layer_norm::CanPackAs<STORE>(store, 2)) {
    return LaunchRmsNormBlockUncachedImpl<LOAD, STORE, ComputeType, 2>(stream, load, store, nrow,
                                                                       ncol, eps, inv_rms);
  } else {
    return LaunchRmsNormBlockUncachedImpl<LOAD, STORE, ComputeType, 1>(stream, load, store, nrow,
                                                                       ncol, eps, inv_rms);
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                   const int64_t nrow, const int64_t ncol,
                                                   const double eps, ComputeType* inv_rms) {
  return DispatchLaunchRmsNormBlockUncachedImplPackSize(stream, load, store, nrow, ncol, eps,
                                                        inv_rms);
}

template<typename LOAD, typename STORE, typename ComputeType>
typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type LaunchRmsNorm(
    cudaStream_t stream, LOAD load, STORE store, const int64_t nrow, const int64_t ncol,
    const double eps, ComputeType* inv_rms) {
  if (ncol <= 1024) {
    return DispatchLaunchRmsNormWarpImpl(stream, load, store, nrow, ncol, eps, inv_rms);
  } else {
    bool dispatch_smem_impl_success = false;
    {
      cudaError_t err = TryDispatchLaunchRmsNormBlockSMemImpl(stream, load, store, nrow, ncol, eps,
                                                              inv_rms, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchLaunchRmsNormBlockUncachedImpl(stream, load, store, nrow, ncol, eps, inv_rms);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType>
typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type LaunchRmsNorm(
    cudaStream_t stream, LOAD load, STORE store, const int64_t nrow, const int64_t ncol,
    const double eps, ComputeType* inv_rms) {
  return DispatchLaunchRmsNormBlockUncachedImpl(stream, load, store, nrow, ncol, eps, inv_rms);
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access>
__global__ void RmsNormGradWarpImpl(const int nrow, const int ncol, LOAD_X load_x, LOAD_DY load_dy,
                                    STORE store, const ComputeType* inv_rms) {
  static_assert(max_cols_per_thread % pack_size == 0, "");
  static_assert(min_cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  assert(ncol <= max_cols_per_thread * thread_group_width);

  constexpr int max_packs = max_cols_per_thread / pack_size;
  constexpr int min_packs = min_cols_per_thread / pack_size;

  ComputeType normalized_buf[rows_per_access][max_cols_per_thread];
  ComputeType dy_buf[rows_per_access][max_cols_per_thread];

  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  for (int row_i = global_thread_group_id; row_i < nrow; row_i += num_global_thread_group) {
    ComputeType sum_stats[rows_per_access];
    ComputeType inv_rms_buf[rows_per_access];
#pragma unroll
    for (int row_j = 0; row_j < rows_per_access; ++row_j) {
      const int global_row = row_i * rows_per_access + row_j;
      sum_stats[row_j] = 0;
      inv_rms_buf[row_j] = inv_rms[global_row];
      ComputeType* row_normalized_buf = normalized_buf[row_j];
      ComputeType* row_dy_buf = dy_buf[row_j];
#pragma unroll
      for (int pack_i = 0; pack_i < min_packs; ++pack_i) {
        const int pack_offset = pack_i * pack_size;
        const int global_col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        load_x.template load<pack_size>(row_normalized_buf + pack_offset, global_row, global_col);
        load_dy.template load<pack_size>(row_dy_buf + pack_offset, global_row, global_col);
#pragma unroll
        for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
          const int col = pack_offset + pack_j;
          row_normalized_buf[col] = row_normalized_buf[col] * inv_rms_buf[row_j];
          sum_stats[row_j] += row_dy_buf[col] * row_normalized_buf[col];
        }
      }
#pragma unroll
      for (int pack_i = min_packs; pack_i < max_packs; ++pack_i) {
        const int pack_offset = pack_i * pack_size;
        const int global_col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        if (global_col < ncol) {
          load_x.template load<pack_size>(row_normalized_buf + pack_offset, global_row, global_col);
          load_dy.template load<pack_size>(row_dy_buf + pack_offset, global_row, global_col);
#pragma unroll
          for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
            const int col = pack_offset + pack_j;
            row_normalized_buf[col] = row_normalized_buf[col] * inv_rms_buf[row_j];
            sum_stats[row_j] += row_dy_buf[col] * row_normalized_buf[col];
          }
        }
      }
    }
    ComputeType warp_sum_stats[rows_per_access];
#pragma unroll
    for (int row_j = 0; row_j < rows_per_access; ++row_j) {
      warp_sum_stats[row_j] =
          layer_norm::WarpAllReduce<layer_norm::SumOp, ComputeType, thread_group_width>(
              sum_stats[row_j]);
    }
#pragma unroll
    for (int row_j = 0; row_j < rows_per_access; ++row_j) {
      const int global_row = row_i * rows_per_access + row_j;
      ComputeType* row_normalized_buf = normalized_buf[row_j];
      ComputeType* row_dy_buf = dy_buf[row_j];
#pragma unroll
      for (int pack_i = 0; pack_i < min_packs; ++pack_i) {
        const int pack_offset = pack_i * pack_size;
        const int global_col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
          const int col = pack_offset + pack_j;
          const ComputeType norm_val =
              layer_norm::Div(row_normalized_buf[col], static_cast<ComputeType>(ncol));
          row_dy_buf[col] =
              (row_dy_buf[col] - norm_val * warp_sum_stats[row_j]) * inv_rms_buf[row_j];
        }
        store.template store<pack_size>(row_dy_buf + pack_offset, global_row, global_col);
      }
#pragma unroll
      for (int pack_i = min_packs; pack_i < max_packs; ++pack_i) {
        const int pack_offset = pack_i * pack_size;
        const int global_col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        if (global_col < ncol) {
          for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
            const int col = pack_offset + pack_j;
            const ComputeType norm_val =
                layer_norm::Div(row_normalized_buf[col], static_cast<ComputeType>(ncol));
            row_dy_buf[col] =
                (row_dy_buf[col] - norm_val * warp_sum_stats[row_j]) * inv_rms_buf[row_j];
          }
          store.template store<pack_size>(row_dy_buf + pack_offset, global_row, global_col);
        }
      }
    }
  }
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access>
cudaError_t LaunchRmsNormGradWarpImpl(cudaStream_t stream, const int nrow, const int ncol,
                                      LOAD_X load_x, LOAD_DY load_dy, STORE store,
                                      const ComputeType* inv_rms) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  const int64_t num_blocks =
      (nrow / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err = layer_norm::GetNumBlocks(
        RmsNormGradWarpImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, max_cols_per_thread,
                            min_cols_per_thread, thread_group_width, rows_per_access>,
        block_size, 0, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  RmsNormGradWarpImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, max_cols_per_thread,
                      min_cols_per_thread, thread_group_width, rows_per_access>
      <<<grid_dim_x, block_dim, 0, stream>>>(nrow, ncol, load_x, load_dy, store, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLaunchRmsNormGradWarpImplCols(
    cudaStream_t stream, const int64_t nrow, const int64_t ncol, LOAD_X load_x, LOAD_DY load_dy,
    STORE store, const ComputeType* inv_rms) {
  if (ncol <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                       \
  else if (ncol <= (thread_group_width)*pack_size) {                                              \
    if (nrow % 2 == 0) {                                                                          \
      return LaunchRmsNormGradWarpImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, pack_size, \
                                       0, thread_group_width, 2>(stream, nrow, ncol, load_x,      \
                                                                 load_dy, store, inv_rms);        \
    } else {                                                                                      \
      return LaunchRmsNormGradWarpImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, pack_size, \
                                       0, thread_group_width, 1>(stream, nrow, ncol, load_x,      \
                                                                 load_dy, store, inv_rms);        \
    }                                                                                             \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col)                                                        \
  else if (ncol <= (max_col)*kWarpSize) {                                                        \
    return LaunchRmsNormGradWarpImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, max_col,    \
                                     min_col, kWarpSize, 1>(stream, nrow, ncol, load_x, load_dy, \
                                                            store, inv_rms);                     \
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

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormGradWarpImplPackSize(cudaStream_t stream, const int64_t nrow,
                                                      const int64_t ncol, LOAD_X load_x,
                                                      LOAD_DY load_dy, STORE store,
                                                      const ComputeType* inv_rms) {
  return DispatchLaunchRmsNormGradWarpImplCols<LOAD_X, LOAD_DY, STORE, ComputeType, 1>(
      stream, nrow, ncol, load_x, load_dy, store, inv_rms);
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size>
__global__ void RmsNormGradBlockSMemImpl(const int nrow, const int ncol, LOAD_X load_x,
                                         LOAD_DY load_dy, STORE store, const ComputeType* inv_rms) {
  extern __shared__ __align__(sizeof(double)) unsigned char dyn_smem[];
  // dynamic shared memory for caching x and dy
  auto* normalized_buf = reinterpret_cast<ComputeType*>(dyn_smem);
  auto* dy_buf = normalized_buf + ncol;
  assert(ncol % pack_size == 0);
  const int num_packs = ncol / pack_size;
  for (int row = blockIdx.x; row < nrow; row += gridDim.x) {
    ComputeType sum_stats = 0;
    const ComputeType inv_rms_val = inv_rms[row];
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += blockDim.x) {
      ComputeType x_pack[pack_size];
      ComputeType dy_pack[pack_size];
      const int pack_offset = pack_i * pack_size;
      load_x.template load<pack_size>(x_pack, row, pack_offset);
      load_dy.template load<pack_size>(dy_pack, row, pack_offset);
#pragma unroll
      for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
        const int col = pack_offset + pack_j;
        normalized_buf[col] = x_pack[pack_j] * inv_rms_val;
        dy_buf[col] = dy_pack[pack_j];
        sum_stats += dy_buf[col] * normalized_buf[col];
      }
    }
    const ComputeType row_sum_stats =
        layer_norm::BlockAllReduce<layer_norm::SumOp, ComputeType, block_size>(sum_stats);
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += blockDim.x) {
      ComputeType pack[pack_size];
      const int pack_offset = pack_i * pack_size;
#pragma unroll
      for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
        const int col = pack_offset + pack_j;
        const ComputeType norm_val =
            layer_norm::Div(normalized_buf[col], static_cast<ComputeType>(ncol));
        pack[pack_j] = (dy_buf[col] - norm_val * row_sum_stats) * inv_rms_val;
      }
      store.template store<pack_size>(pack, row, pack_offset);
    }
  }
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size>
cudaError_t LaunchRmsNormGradBlockSMemImpl(cudaStream_t stream, const int64_t nrow,
                                           const int64_t ncol, const size_t smem_size,
                                           LOAD_X load_x, LOAD_DY load_dy, STORE store,
                                           const ComputeType* inv_rms) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = layer_norm::GetNumBlocks(
        RmsNormGradBlockSMemImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, block_size>,
        block_size, smem_size, nrow, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  RmsNormGradBlockSMemImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem_size, stream>>>(
          static_cast<int>(nrow), static_cast<int>(ncol), load_x, load_dy, store, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size>
cudaError_t TryDispatchLaunchRmsNormGradBlockSMemImplBlockSize(
    cudaStream_t stream, const int64_t nrow, const int64_t ncol, LOAD_X load_x, LOAD_DY load_dy,
    STORE store, const ComputeType* inv_rms, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem_size = ncol * sizeof(ComputeType) * 2;  // ncol * 2 for caching x and dy both
  int max_active_blocks = 0;
  int num_blocks = 0;

#define SELECT_BLOCK_SIZE_CONF(block_size_conf)                                                    \
  {                                                                                                \
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(                               \
        &num_blocks,                                                                               \
        RmsNormGradBlockSMemImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, block_size_conf>, \
        block_size_conf, smem_size);                                                               \
    if (err != cudaSuccess) { return err; }                                                        \
    if (max_active_blocks == 0) {                                                                  \
      if (num_blocks <= max_active_blocks) {                                                       \
        *success = false;                                                                          \
        return cudaSuccess;                                                                        \
      }                                                                                            \
      max_active_blocks = num_blocks;                                                              \
    } else {                                                                                       \
      if (num_blocks == max_active_blocks) {                                                       \
        *success = true;                                                                           \
        return LaunchRmsNormGradBlockSMemImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size,      \
                                              block_size_conf>(stream, nrow, ncol, smem_size,      \
                                                               load_x, load_dy, store, inv_rms);   \
      }                                                                                            \
    }                                                                                              \
  }

  SELECT_BLOCK_SIZE_CONF(block_size_conf_1)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_4)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_3)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_2)
#undef SELECT_BLOCK_SIZE_CONF

  *success = true;
  return LaunchRmsNormGradBlockSMemImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size,
                                        block_size_conf_1>(stream, nrow, ncol, smem_size, load_x,
                                                           load_dy, store, inv_rms);
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType>
cudaError_t TryDispatchLaunchRmsNormGradBlockSMemImplPackSize(
    cudaStream_t stream, const int64_t nrow, const int64_t ncol, LOAD_X load_x, LOAD_DY load_dy,
    STORE store, const ComputeType* inv_rms, bool* success) {
  if (ncol % 2 == 0 && layer_norm::CanPackAs<LOAD_X>(load_x, 2)
      && layer_norm::CanPackAs<LOAD_DY>(load_dy, 2) && layer_norm::CanPackAs<STORE>(store, 2)) {
    return TryDispatchLaunchRmsNormGradBlockSMemImplBlockSize<LOAD_X, LOAD_DY, STORE, ComputeType,
                                                              2>(stream, nrow, ncol, load_x,
                                                                 load_dy, store, inv_rms, success);
  } else {
    return TryDispatchLaunchRmsNormGradBlockSMemImplBlockSize<LOAD_X, LOAD_DY, STORE, ComputeType,
                                                              1>(stream, nrow, ncol, load_x,
                                                                 load_dy, store, inv_rms, success);
  }
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size>
__global__ void RmsNormGradBlockUncachedImpl(const int nrow, const int ncol, LOAD_X load_x,
                                             LOAD_DY load_dy, STORE store,
                                             const ComputeType* inv_rms) {
  assert(ncol % pack_size == 0);
  const int num_packs = ncol / pack_size;
  for (int row = blockIdx.x; row < nrow; row += gridDim.x) {
    const ComputeType inv_rms_val = inv_rms[row];
    ComputeType sum_stats = 0;
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += blockDim.x) {
      ComputeType x_pack[pack_size];
      ComputeType dy_pack[pack_size];
      const int pack_offset = pack_i * pack_size;
      load_x.template load<pack_size>(x_pack, row, pack_offset);
      load_dy.template load<pack_size>(dy_pack, row, pack_offset);
#pragma unroll
      for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
        sum_stats += dy_pack[pack_j] * x_pack[pack_j] * inv_rms_val;
      }
    }
    const ComputeType row_sum_stats =
        layer_norm::BlockAllReduce<layer_norm::SumOp, ComputeType, block_size>(sum_stats);
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += blockDim.x) {
      ComputeType x_pack[pack_size];
      ComputeType dy_pack[pack_size];
      const int pack_offset = pack_i * pack_size;
      load_x.template load<pack_size>(x_pack, row, pack_offset);
      load_dy.template load<pack_size>(dy_pack, row, pack_offset);
#pragma unroll
      for (int pack_j = 0; pack_j < pack_size; ++pack_j) {
        const ComputeType norm_val =
            layer_norm::Div(x_pack[pack_j] * inv_rms_val, static_cast<ComputeType>(ncol));
        dy_pack[pack_j] = (dy_pack[pack_j] - norm_val * row_sum_stats) * inv_rms_val;
      }
      store.template store<pack_size>(dy_pack, row, pack_offset);
    }
  }
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size,
         int block_size>
cudaError_t LaunchRmsNormGradBlockUncachedImpl(cudaStream_t stream, const int64_t nrow,
                                               const int64_t ncol, LOAD_X load_x, LOAD_DY load_dy,
                                               STORE store, const ComputeType* inv_rms) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = layer_norm::GetNumBlocks(
        RmsNormGradBlockUncachedImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, block_size>,
        block_size, 0, nrow, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  RmsNormGradBlockUncachedImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(nrow, ncol, load_x, load_dy, store, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType, int pack_size>
cudaError_t DispatchLaunchRmsNormGradBlockUncachedImplBlockSize(cudaStream_t stream,
                                                                const int64_t nrow,
                                                                const int64_t ncol, LOAD_X load_x,
                                                                LOAD_DY load_dy, STORE store,
                                                                const ComputeType* inv_rms) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  int max_active_blocks = 0;

#define SELECT_BLOCK_SIZE_CONF(block_size_conf)                                                 \
  {                                                                                             \
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(                            \
        &max_active_blocks,                                                                     \
        RmsNormGradBlockUncachedImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size,            \
                                     block_size_conf>,                                          \
        block_size_conf, 0);                                                                    \
    if (err != cudaSuccess) { return err; }                                                     \
    if (max_active_blocks > 0) {                                                                \
      return LaunchRmsNormGradBlockUncachedImpl<LOAD_X, LOAD_DY, STORE, ComputeType, pack_size, \
                                                block_size_conf>(stream, nrow, ncol, load_x,    \
                                                                 load_dy, store, inv_rms);      \
    }                                                                                           \
  }

  SELECT_BLOCK_SIZE_CONF(block_size_conf_4)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_3)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_2)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_1)
#undef SELECT_BLOCK_SIZE_CONF

  return cudaErrorInvalidValue;
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormGradBlockUncachedImplPackSize(cudaStream_t stream,
                                                               const int64_t nrow,
                                                               const int64_t ncol, LOAD_X load_x,
                                                               LOAD_DY load_dy, STORE store,
                                                               const ComputeType* inv_rms) {
  if (ncol % 2 == 0 && layer_norm::CanPackAs<LOAD_X>(load_x, 2)
      && layer_norm::CanPackAs<LOAD_DY>(load_dy, 2) && layer_norm::CanPackAs<STORE>(store, 2)
      && ncol > kWarpSize) {
    return DispatchLaunchRmsNormGradBlockUncachedImplBlockSize<LOAD_X, LOAD_DY, STORE, ComputeType,
                                                               2>(stream, nrow, ncol, load_x,
                                                                  load_dy, store, inv_rms);
  } else {
    return DispatchLaunchRmsNormGradBlockUncachedImplBlockSize<LOAD_X, LOAD_DY, STORE, ComputeType,
                                                               1>(stream, nrow, ncol, load_x,
                                                                  load_dy, store, inv_rms);
  }
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType>
typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
LaunchRmsNormGrad(cudaStream_t stream, const int64_t nrow, const int64_t ncol, LOAD_X load_x,
                  LOAD_DY load_dy, STORE store, const ComputeType* inv_rms) {
  if (ncol <= 1024) {
    return DispatchLaunchRmsNormGradWarpImplPackSize(stream, nrow, ncol, load_x, load_dy, store,
                                                     inv_rms);
  } else {
    bool dispatch_smem_impl_success = false;
    {
      cudaError_t err = TryDispatchLaunchRmsNormGradBlockSMemImplPackSize(
          stream, nrow, ncol, load_x, load_dy, store, inv_rms, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchLaunchRmsNormGradBlockUncachedImplPackSize(stream, nrow, ncol, load_x, load_dy,
                                                                store, inv_rms);
    }
    return cudaSuccess;
  }
}

template<typename LOAD_X, typename LOAD_DY, typename STORE, typename ComputeType>
typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
LaunchRmsNormGrad(cudaStream_t stream, const int64_t nrow, const int64_t ncol, LOAD_X load_x,
                  LOAD_DY load_dy, STORE store, const ComputeType* inv_rms) {
  return DispatchLaunchRmsNormGradBlockUncachedImplPackSize(stream, nrow, ncol, load_x, load_dy,
                                                            store, inv_rms);
}

template<int nproc_per_thread, typename T, typename ComputeType>
__global__ void RmsNormParamGrad(int nrow, int ncol, const T* __restrict__ dy,
                                 const T* __restrict__ x, const ComputeType* __restrict__ inv_rms,
                                 T* __restrict__ b_weight_grad) {
  __shared__ ComputeType dweight[kWarpSize][kWarpSize + 1];
  ComputeType dweight_sum[nproc_per_thread];
#pragma unroll
  for (int i = 0; i < nproc_per_thread; ++i) { dweight_sum[i] = 0; }
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < ncol) {
    // a wave for one traverse (when nrow > warp_size * grad_dim_y)
    for (int j = blockIdx.y * kWarpSize + threadIdx.y; j < nrow; j += kWarpSize * gridDim.y) {
#pragma unroll
      for (int i = 0; i < nproc_per_thread; ++i) {
        int row = j + i * blockDim.y;
        if (row < nrow) {
          int offset = row * ncol + col;
          const ComputeType dy_val = static_cast<ComputeType>(dy[offset]);
          const ComputeType x_val = static_cast<ComputeType>(x[offset]);
          const ComputeType inv_rms_val = inv_rms[row];
          // collect dx from waves
          dweight_sum[i] += dy_val * x_val * inv_rms_val;
        }
      }
    }
  }
  // broadcast sum to the nproc_per_thread number rows
  // each warp process the nproc_per_thread number rows of smem
#pragma unroll
  for (int i = 0; i < nproc_per_thread; ++i) {
    dweight[i * blockDim.y + threadIdx.y][threadIdx.x] = dweight_sum[i];
  }
  __syncthreads();
  // transpose access for leveraging warp to reduce rows in a block
#pragma unroll
  for (int i = 0; i < nproc_per_thread; ++i) {
    // the first col of block threads is for storing the reduced sum of rows,
    // and each first col thread is writing the nproc_per_thread number cols of output
    const int row_in_block = threadIdx.y + i * blockDim.y;
    const int col = blockIdx.x * blockDim.x + row_in_block;
    if (col < ncol) {
      // each warp process a col in which reduce sum all rows
      ComputeType dweight_val = dweight[threadIdx.x][row_in_block];
      ComputeType global_dweight = WarpReduceSum<ComputeType>(dweight_val);
      if (threadIdx.x == 0) {
        const int offset = blockIdx.y * ncol + col;
        b_weight_grad[offset] = global_dweight;
      }
    }
  }
}

template<int nproc_per_thread, typename T>
cudaError_t GetGrid2Dim(const int64_t nrow, const int64_t ncol, int block_dim_x, int block_dim_y,
                        int* grid_dim_x, int* grid_dim_y) {
  const int tile_size = block_dim_x;
  if (nproc_per_thread * block_dim_y != tile_size) { return cudaErrorInvalidValue; }
  *grid_dim_x = (ncol + tile_size - 1) / tile_size;
  const int num_blocks_y = (nrow + tile_size - 1) / tile_size;

  using ComputeType = typename layer_norm::DefaultComputeType<T>::type;
  cudaError_t err = layer_norm::GetNumBlocks(RmsNormParamGrad<nproc_per_thread, T, ComputeType>,
                                             block_dim_x * block_dim_y, /*dynamic_smem_size*/ 0,
                                             num_blocks_y, /*waves*/ 1, grid_dim_y);
  if (err != cudaSuccess) { return err; }
  return cudaSuccess;
}

}  // namespace rms_norm
}  // namespace cuda
}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_RMS_NORM_H_
