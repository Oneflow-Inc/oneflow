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

#ifndef ONEFLOW_CORE_CUDA_RMS_NORM_OUTPUT_NORM_ARG_H_
#define ONEFLOW_CORE_CUDA_RMS_NORM_OUTPUT_NORM_ARG_H_

#include "oneflow/core/cuda/layer_norm.cuh"

namespace oneflow {
namespace cuda {
namespace rms_norm_output_norm_arg {

constexpr int kWarpSize = 32;

template<typename T>
__inline__ __device__ T WarpReduceSum(T val) {
  for (int mask = 16; mask > 0; mask /= 2) { val += __shfl_down_sync(0xffffffff, val, mask); }
  return val;
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access, bool padding>
__global__ void RmsNormOutputNormArgWarpImpl(LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int nrow, const int ncol,
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
        norm_arg_store.template store<pack_size>(row_buf + pack_offset, row, col);
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
          norm_arg_store.template store<pack_size>(row_buf + pack_offset, row, col);
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
        out_put_store.template store<pack_size>(row_buf + pack_i * pack_size, row, col);
      }
#pragma unroll
      for (int pack_i = min_packs; pack_i < max_packs; ++pack_i) {
        const int col = (pack_i * thread_group_width + threadIdx.x) * pack_size;
        if (!padding || col < ncol) {
          out_put_store.template store<pack_size>(row_buf + pack_i * pack_size, row, col);
        }
      }
    }
  }
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access, bool padding>
cudaError_t LaunchRmsNormOutputNormArgWarpImpl(cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int64_t nrow,
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
        RmsNormOutputNormArgWarpImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, max_cols_per_thread,
                        min_cols_per_thread, thread_group_width, rows_per_access, padding>,
        block_size, 0, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  RmsNormOutputNormArgWarpImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, max_cols_per_thread, min_cols_per_thread,
                  thread_group_width, rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, norm_arg_store, out_put_store, static_cast<int>(nrow),
                                             static_cast<int>(ncol), eps, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size,
         int max_cols_per_thread, int min_cols_per_thread, int thread_group_width,
         int rows_per_access>
cudaError_t DispatchLaunchRmsNormOutputNormArgWarpImplPadding(cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store,
                                                 const int64_t nrow, const int64_t ncol,
                                                 const double eps, ComputeType* inv_rms) {
  if (ncol == max_cols_per_thread * thread_group_width) {
    // when not padding, min_cols_per_thread must equals to max_cols_per_thread, pass
    // max_cols_per_thread as min_cols_per_thread and max_cols_per_thread param.
    return LaunchRmsNormOutputNormArgWarpImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, max_cols_per_thread,
                                 max_cols_per_thread, thread_group_width, rows_per_access, false>(
        stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);
  } else {
    return LaunchRmsNormOutputNormArgWarpImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, max_cols_per_thread,
                                 min_cols_per_thread, thread_group_width, rows_per_access, true>(
        stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);
  }
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLaunchRmsNormOutputNormArgWarpImplCols(
    cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int64_t nrow, const int64_t ncol,
    const double eps, ComputeType* inv_rms) {
  if (ncol <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                       \
  else if (ncol <= (thread_group_width)*pack_size) {                                              \
    if (nrow % 2 == 0) {                                                                          \
      return DispatchLaunchRmsNormOutputNormArgWarpImplPadding<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, pack_size, \
                                                  0, thread_group_width, 2>(                      \
          stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);                                         \
    } else {                                                                                      \
      return DispatchLaunchRmsNormOutputNormArgWarpImplPadding<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, pack_size, \
                                                  0, thread_group_width, 1>(                      \
          stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);                                         \
    }                                                                                             \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col)                                                         \
  else if (ncol <= (max_col)*kWarpSize) {                                                         \
    return DispatchLaunchRmsNormOutputNormArgWarpImplPadding<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, max_col,     \
                                                min_col, kWarpSize, 1>(stream, load, norm_arg_store, out_put_store, nrow, \
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

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchLaunchRmsNormOutputNormArgWarpImplCols(
    cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int64_t nrow, const int64_t ncol,
    const double eps, ComputeType* inv_rms) {
  if (ncol <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                       \
  else if (ncol <= (thread_group_width)*pack_size) {                                              \
    if (nrow % 2 == 0) {                                                                          \
      return DispatchLaunchRmsNormOutputNormArgWarpImplPadding<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, pack_size, \
                                                  0, thread_group_width, 2>(                      \
          stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);                                         \
    } else {                                                                                      \
      return DispatchLaunchRmsNormOutputNormArgWarpImplPadding<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, pack_size, \
                                                  0, thread_group_width, 1>(                      \
          stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);                                         \
    }                                                                                             \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(max_col, min_col)                                                         \
  else if ((ncol <= (max_col)*kWarpSize) && (ncol > (min_col)*kWarpSize)) {                       \
    return DispatchLaunchRmsNormOutputNormArgWarpImplPadding<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, max_col,     \
                                                min_col, kWarpSize, 1>(stream, load, norm_arg_store, out_put_store, nrow, \
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

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormOutputNormArgWarpImplPackSize(cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store,
                                                  const int64_t nrow, const int64_t ncol,
                                                  const double eps, ComputeType* inv_rms) {
  if (ncol % 2 == 0 && layer_norm::CanPackAs<LOAD>(load, 2)
      && layer_norm::CanPackAs<OUTPUT_STORE>(out_put_store, 2)) {
    return DispatchLaunchRmsNormOutputNormArgWarpImplCols<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, 2>(stream, load, norm_arg_store, out_put_store, nrow,
                                                                          ncol, eps, inv_rms);
  } else {
    return DispatchLaunchRmsNormOutputNormArgWarpImplCols<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, 1>(stream, load, norm_arg_store, out_put_store, nrow,
                                                                          ncol, eps, inv_rms);
  }
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormOutputNormArgWarpImpl(cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store,
                                          const int64_t nrow, const int64_t ncol, const double eps,
                                          ComputeType* inv_rms) {
  return DispatchLaunchRmsNormOutputNormArgWarpImplPackSize(stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size, int block_size>
__global__ void RmsNormOutputNormArgBlockSMemImpl(LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int nrow, const int ncol,
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
      norm_arg_store.template store<pack_size>(pack, row, col);
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
      out_put_store.template store<pack_size>(pack, row, col);
    }
  }
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size, int block_size>
cudaError_t LaunchRmsNormOutputNormArgBlockSMemImpl(cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store,
                                       size_t smem_size, const int64_t nrow, const int64_t ncol,
                                       const double eps, ComputeType* inv_rms) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = layer_norm::GetNumBlocks(
        RmsNormOutputNormArgBlockSMemImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, block_size>, block_size,
        smem_size, nrow, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  RmsNormOutputNormArgBlockSMemImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem_size, stream>>>(load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size>
cudaError_t TryDispatchLaunchRmsNormOutputNormArgBlockSMemImplBlockSize(cudaStream_t stream, LOAD load,
                                                           NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int64_t nrow,
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
        &num_blocks, RmsNormOutputNormArgBlockSMemImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, block_size_conf>, \
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
        return LaunchRmsNormOutputNormArgBlockSMemImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, block_size_conf>( \
            stream, load, norm_arg_store, out_put_store, smem_size, nrow, ncol, eps, inv_rms);                           \
      }                                                                                          \
    }                                                                                            \
  }

  SELECT_BLOCK_SIZE_CONF(block_size_conf_1)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_4)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_3)
  SELECT_BLOCK_SIZE_CONF(block_size_conf_2)
#undef SELECT_BLOCK_SIZE_CONF

  *success = true;
  return LaunchRmsNormOutputNormArgBlockSMemImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, block_size_conf_1>(
      stream, load, norm_arg_store, out_put_store, smem_size, nrow, ncol, eps, inv_rms);
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType>
cudaError_t TryDispatchLaunchRmsNormOutputNormArgBlockSMemImplPackSize(cudaStream_t stream, LOAD load,
                                                          NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int64_t nrow,
                                                          const int64_t ncol, const double eps,
                                                          ComputeType* inv_rms, bool* success) {
  if (ncol % 4 == 0 && layer_norm::CanPackAs<LOAD>(load, 4)
      && layer_norm::CanPackAs<OUTPUT_STORE>(out_put_store, 4)) {
    return TryDispatchLaunchRmsNormOutputNormArgBlockSMemImplBlockSize<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, 4>(
        stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms, success);
  } else if (ncol % 2 == 0 && layer_norm::CanPackAs<LOAD>(load, 2)
             && layer_norm::CanPackAs<OUTPUT_STORE>(out_put_store, 2)) {
    return TryDispatchLaunchRmsNormOutputNormArgBlockSMemImplBlockSize<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, 2>(
        stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms, success);
  } else {
    return TryDispatchLaunchRmsNormOutputNormArgBlockSMemImplBlockSize<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, 1>(
        stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms, success);
  }
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType>
cudaError_t TryDispatchLaunchRmsNormOutputNormArgBlockSMemImpl(cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store,
                                                  const int64_t nrow, const int64_t ncol,
                                                  const double eps, ComputeType* inv_rms,
                                                  bool* success) {
  return TryDispatchLaunchRmsNormOutputNormArgBlockSMemImplPackSize(stream, load, norm_arg_store, out_put_store, nrow, ncol, eps,
                                                       inv_rms, success);
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size, int block_size>
__global__ void RmsNormOutputNormArgBlockUncachedImpl(LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int nrow, const int ncol,
                                         const double eps, ComputeType* inv_rms) {
  assert(ncol % pack_size == 0);
  const int num_packs = ncol / pack_size;
  for (int row = blockIdx.x; row < nrow; row += gridDim.x) {
    ComputeType thread_square_sum = 0;
    for (int pack_i = threadIdx.x; pack_i < num_packs; pack_i += block_size) {
      ComputeType pack[pack_size];
      const int col = pack_i * pack_size;
      load.template load<pack_size>(pack, row, col);
      norm_arg_store.template store<pack_size>(pack, row, col);
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
      out_put_store.template store<pack_size>(pack, row, col);
    }
  }
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType, int pack_size>
cudaError_t LaunchRmsNormOutputNormArgBlockUncachedImpl(cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store,
                                           const int64_t nrow, const int64_t ncol, const double eps,
                                           ComputeType* inv_rms) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = layer_norm::GetNumBlocks(
        RmsNormOutputNormArgBlockUncachedImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, block_size>, block_size, 0,
        nrow, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  RmsNormOutputNormArgBlockUncachedImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormOutputNormArgBlockUncachedImplPackSize(cudaStream_t stream, LOAD load,
                                                           NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int64_t nrow,
                                                           const int64_t ncol, const double eps,
                                                           ComputeType* inv_rms) {
  if (ncol % 4 == 0 && layer_norm::CanPackAs<LOAD>(load, 4)
      && layer_norm::CanPackAs<OUTPUT_STORE>(out_put_store, 4)) {
    return LaunchRmsNormOutputNormArgBlockUncachedImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, 4>(stream, load, norm_arg_store, out_put_store, nrow,
                                                                       ncol, eps, inv_rms);
  } else if (ncol % 2 == 0 && layer_norm::CanPackAs<LOAD>(load, 2)
             && layer_norm::CanPackAs<OUTPUT_STORE>(out_put_store, 2)) {
    return LaunchRmsNormOutputNormArgBlockUncachedImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, 2>(stream, load, norm_arg_store, out_put_store, nrow,
                                                                       ncol, eps, inv_rms);
  } else {
    return LaunchRmsNormOutputNormArgBlockUncachedImpl<LOAD, NORM_ARG_STORE, OUTPUT_STORE, ComputeType, 1>(stream, load, norm_arg_store, out_put_store, nrow,
                                                                       ncol, eps, inv_rms);
  }
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType>
cudaError_t DispatchLaunchRmsNormOutputNormArgBlockUncachedImpl(cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store,
                                                   const int64_t nrow, const int64_t ncol,
                                                   const double eps, ComputeType* inv_rms) {
  return DispatchLaunchRmsNormOutputNormArgBlockUncachedImplPackSize(stream, load, norm_arg_store, out_put_store, nrow, ncol, eps,
                                                        inv_rms);
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType>
typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type LaunchRmsNormOutputNormArg(
    cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int64_t nrow, const int64_t ncol,
    const double eps, ComputeType* inv_rms) {
  if (ncol <= 1024) {
    return DispatchLaunchRmsNormOutputNormArgWarpImpl(stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);
  } else {
    bool dispatch_smem_impl_success = false;
    {
      cudaError_t err = TryDispatchLaunchRmsNormOutputNormArgBlockSMemImpl(stream, load, norm_arg_store, out_put_store, nrow, ncol, eps,
                                                              inv_rms, &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchLaunchRmsNormOutputNormArgBlockUncachedImpl(stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename NORM_ARG_STORE, typename OUTPUT_STORE, typename ComputeType>
typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type LaunchRmsNormOutputNormArg(
    cudaStream_t stream, LOAD load, NORM_ARG_STORE norm_arg_store, OUTPUT_STORE out_put_store, const int64_t nrow, const int64_t ncol,
    const double eps, ComputeType* inv_rms) {
  return DispatchLaunchRmsNormOutputNormArgBlockUncachedImpl(stream, load, norm_arg_store, out_put_store, nrow, ncol, eps, inv_rms);
}

}  // namespace rms_norm_output_norm_arg
}  // namespace cuda
}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_RMS_NORM_OUTPUT_NORM_ARG_H_
