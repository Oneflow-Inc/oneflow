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
#include "oneflow/core/primitive/include/permute.h"
#include "oneflow/core/primitive/common/permute_impl.h"
#include "oneflow/core/stream/cuda/cuda_stream_context.h"
#include <cuda_runtime.h>

namespace oneflow {

namespace primitive {

namespace permute {

namespace internal {

namespace {

constexpr int32_t kMov4TileSize = 32;
constexpr int32_t kMov2TileSize = 64;
constexpr int32_t kBlockRows = 8;

template<size_t num_dims, size_t movement_size, typename IndexType>
__global__ void PermuteKernel(PermuteKernelParams<num_dims, IndexType> params) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  const T* src = reinterpret_cast<const T*>(params.src);
  T* dst = reinterpret_cast<T*>(params.dst);
  IndexType src_index[num_dims];
  IndexType dst_index[num_dims];
  CUDA_1D_KERNEL_LOOP_T(IndexType, i, params.count) {
    params.dst_index_helper.OffsetToNdIndex(i, dst_index);
#pragma unroll
    for (size_t dim = 0; dim < num_dims; ++dim) {
      src_index[params.permutation[dim]] = dst_index[dim];
    }
    IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
    dst[i] = src[src_offset];
  }
}

// (B, X, Y) -> (B, Y, X)
// refer from https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template<size_t num_dims, size_t movement_size, size_t tile_size, typename IndexType>
__global__ void BatchTransposeKernel(const void* src_ptr, void* dst_ptr, IndexType H, IndexType W,
                                     IndexType num_tile_rows, IndexType num_tile_cols,
                                     int32_t block_nums) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  __shared__ T tile[tile_size][tile_size + 1];  // To avoid bank conflict.

  const T* src = reinterpret_cast<const T*>(src_ptr);
  T* dst = reinterpret_cast<T*>(dst_ptr);

  IndexType batch_num_tile = num_tile_rows * num_tile_cols;
  for (int i = blockIdx.x, step = gridDim.x; i < block_nums; i += step) {
    const IndexType batch_index = i / batch_num_tile;  // the index of batch.
    const IndexType flatten_index =
        i - batch_index * batch_num_tile;  // equal to i % (num_tile_rows*num_tile_cols). the
                                           // flatten index of tile in a batch.

    const IndexType row_index = flatten_index / num_tile_cols;  // the row index of tile in a batch.
    const IndexType col_index =
        flatten_index
        - row_index
              * num_tile_cols;  // equal to k % num_tile_cols. the col index of tile in a batch.
    const IndexType offset = batch_index * H * W;
    IndexType x = col_index * tile_size + threadIdx.x;
    IndexType y = row_index * tile_size + threadIdx.y;
    if (x < W) {
      IndexType y_range =
          ((tile_size - threadIdx.y) < (H - y)) ? (tile_size - threadIdx.y) : (H - y);
#pragma unroll
      // each thread process 4 elements.
      // `i < y_range` equals to: `threadIdx.y + i < tile_size && y + i < H`.
      for (int i = 0; i < y_range; i += kBlockRows) {
        tile[threadIdx.y + i][threadIdx.x] = src[offset + (y + i) * W + x];
      }
    }
    __syncthreads();
    x = row_index * tile_size + threadIdx.x;
    y = col_index * tile_size + threadIdx.y;
    if (x < H) {
      IndexType x_range =
          ((tile_size - threadIdx.y) < (W - y)) ? (tile_size - threadIdx.y) : (W - y);
#pragma unroll
      // `i < x_range` equals to: `threadIdx.y + i < tile_size && y + i < W`.
      for (int i = 0; i < x_range; i += kBlockRows) {
        dst[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
      }
    }
    __syncthreads();
  }
}

/*
Here is a Movementsie=2 version of Batch Transpose.
When the H W can be divided by 2. we can read data use movementsize=4, and write back as
movementsize=2.
*/
template<size_t num_dims, size_t tile_size, typename IndexType>
__global__ void BatchTransposeMovement2Kernel(const void* src_ptr, void* dst_ptr, IndexType rows,
                                              IndexType cols, IndexType num_tile_rows,
                                              IndexType num_tile_cols, int32_t block_nums) {
  static_assert(tile_size % 2 == 0, "");
  using T_MOV2 = typename std::aligned_storage<2, 2>::type;
  using T_MOV4 = typename std::aligned_storage<4, 4>::type;

  const T_MOV4* src = reinterpret_cast<const T_MOV4*>(src_ptr);
  T_MOV4* dst = reinterpret_cast<T_MOV4*>(dst_ptr);

  // Use union structure to process Load and Store.
  __shared__ union {
    T_MOV2 tile_m2[tile_size][tile_size + 2];      // half [64][66]
    T_MOV4 tile_m4[tile_size][tile_size / 2 + 1];  // half2 [64][33]
  } tile_mem;

  IndexType batch_num_tile = num_tile_rows * num_tile_cols;
  for (int i = blockIdx.x, step = gridDim.x; i < block_nums; i += step) {
    const IndexType batch_index = i / batch_num_tile;  // the index of batch.
    const IndexType flatten_index =
        i - batch_index * batch_num_tile;  // equal to i%(num_tile_rows*num_tile_cols). the flatten
                                           // index of tile in a batch.

    const IndexType row_index = flatten_index / num_tile_cols;  // the row index of tile in a batch.
    const IndexType col_index =
        flatten_index
        - row_index
              * num_tile_cols;  // equal to k % num_tile_cols. the col index of tile in a batch.
    const IndexType offset = batch_index * rows * cols;
    IndexType x =
        col_index * tile_size + threadIdx.x * 2;  // cause each thread process a half2 element, we
                                                  // need to multiply 2 for threadIdx.x.
    IndexType y = row_index * tile_size + threadIdx.y;
    if (x < cols) {
      // each thread process 4 elements.
      IndexType y_range =
          ((tile_size - threadIdx.y) < (rows - y)) ? (tile_size - threadIdx.y) : (rows - y);
#pragma unroll
      // `i < y_range` equals to: `threadIdx.y + i < tile_size && y + i < rows`.
      for (int i = 0; i < y_range; i += kBlockRows) {
        // each thread load a half2.
        tile_mem.tile_m4[threadIdx.y + i][threadIdx.x] = src[(offset + (y + i) * cols + x) / 2];
      }
    }
    __syncthreads();
    x = row_index * tile_size + threadIdx.x * 2;  // cause each thread process a half2 element, we
                                                  // need to multiply 2 for threadIdx.x.
    y = col_index * tile_size + threadIdx.y;
    if (x < rows) {
      IndexType x_range =
          ((tile_size - threadIdx.y) < (cols - y)) ? (tile_size - threadIdx.y) : (cols - y);
#pragma unroll
      // `i < x_range` equals to: `threadIdx.y + i < tile_size && y + i < cols`.
      for (int i = 0; i < x_range; i += kBlockRows) {
        /*
        When write back as column, it cannot be stored as half2 directly.
        So we split as 2 half elements, and write back separately.
        */
        union {
          T_MOV4 m4;
          T_MOV2 m2[2];
        } tmp_storage;
        tmp_storage.m2[0] = tile_mem.tile_m2[threadIdx.x * 2][threadIdx.y + i];
        tmp_storage.m2[1] = tile_mem.tile_m2[threadIdx.x * 2 + 1][threadIdx.y + i];
        dst[(offset + (y + i) * rows + x) / 2] = tmp_storage.m4;
      }
    }
    __syncthreads();
  }
}

template<size_t num_dims, size_t movement_size, size_t tile_size, typename IndexType>
void LaunchBatchTransposeKernel(cudaStream_t& cuda_stream,
                                const PermuteKernelParams<num_dims, IndexType>& params,
                                const IndexType& num_batches, const IndexType& rows,
                                const IndexType& cols) {
  IndexType num_tile_rows = (rows + tile_size - 1) / tile_size;
  IndexType num_tile_cols = (cols + tile_size - 1) / tile_size;

  const int32_t block_nums = num_batches * num_tile_rows * num_tile_cols;
  int32_t checked_block_nums = std::min(block_nums, kCudaMaxBlocksNum);
  if (tile_size == kMov2TileSize) {
    const int32_t half2_thread = tile_size / 2;  // cause each thread process two half elements.
    BatchTransposeMovement2Kernel<num_dims, kMov2TileSize, IndexType>
        <<<checked_block_nums, dim3(half2_thread, kBlockRows), 0, cuda_stream>>>(
            params.src, params.dst, rows, cols, num_tile_rows, num_tile_cols,
            block_nums);  // Set threads num as 32x8 cause each threads
                          // process 4 elements to 32x32 share memory.
  } else {
    BatchTransposeKernel<num_dims, movement_size, tile_size, IndexType>
        <<<checked_block_nums, dim3(tile_size, kBlockRows), 0, cuda_stream>>>(
            params.src, params.dst, rows, cols, num_tile_rows, num_tile_cols, block_nums);
  }
}

template<size_t tile_size, typename IndexType>
bool CheckIfGreaterEqualThanTileSize(const IndexType& rows, const IndexType& cols) {
  if (rows < tile_size || cols < tile_size) { return false; }
  return true;
}

template<size_t num_dims, size_t tile_size, typename IndexType>
bool CheckLaunchBatchTranspose(const int* permutation, const IndexType& num_batches,
                               const IndexType& rows, const IndexType& cols) {
  if (CheckIfGreaterEqualThanTileSize<tile_size, IndexType>(rows, cols)) {
    if (num_batches == 1) {
      // 2d tensor case: (0, 1) -> (1, 0)
      return true;
    } else if (num_dims == 3 && permutation[2] == 1 && permutation[1] == 2) {
      // 3d tensor case: (0, 1, 2) -> (0, 2, 1)
      return true;
    } else {
      return false;
    }
  }
  return false;
}

template<typename IndexType, size_t movement_size>
bool CheckUseMov2(const IndexType& rows, const IndexType& cols, const void* src, void* dst) {
  auto src_ptr = reinterpret_cast<std::uintptr_t>(src);
  auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
  return (movement_size == 2) && (rows % 2 == 0) && (cols % 2 == 0) && (src_ptr % 4 == 0)
         && (dst_ptr % 4 == 0);
  ;
}

template<size_t num_dims, typename IndexType>
void InferBatchTransposeShape(const int64_t* src_dims, IndexType* num_batches, IndexType* rows,
                              IndexType* cols) {
  if (num_dims == 2) {
    *num_batches = 1;
    *rows = src_dims[0];
    *cols = src_dims[1];
  } else {
    *num_batches = src_dims[0];
    *rows = src_dims[1];
    *cols = src_dims[2];
  }
}

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(StreamContext* stream_ctx, const int64_t* src_dims, const void* src,
                  const int* permutation, void* dst, size_t count) {
  PermuteKernelParams<num_dims, IndexType> params =
      MakePermuteParams<num_dims, IndexType>(src_dims, src, permutation, dst, count);
  cudaStream_t cuda_stream =
      CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();

  if (num_dims == 2 || num_dims == 3) {
    IndexType num_batches;
    IndexType rows;
    IndexType cols;
    InferBatchTransposeShape<num_dims, IndexType>(src_dims, &num_batches, &rows, &cols);
    if (CheckLaunchBatchTranspose<num_dims, kMov4TileSize>(params.permutation, num_batches, rows,
                                                           cols)) {
      if (CheckUseMov2<IndexType, movement_size>(rows, cols, src, dst)) {
        LaunchBatchTransposeKernel<num_dims, 2, kMov2TileSize, IndexType>(cuda_stream, params,
                                                                          num_batches, rows, cols);
      } else {
        LaunchBatchTransposeKernel<num_dims, movement_size, kMov4TileSize, IndexType>(
            cuda_stream, params, num_batches, rows, cols);
      }
    } else {
      PermuteKernel<num_dims, movement_size, IndexType>
          <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(params);
    }
  } else {
    PermuteKernel<num_dims, movement_size, IndexType>
        <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(params);
  }
}

class PermuteImpl : public Permute {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteImpl);
  PermuteImpl() = default;
  ~PermuteImpl() override = default;

  using Permute::Launch;
  void Launch(StreamContext* stream_ctx, DataType data_type, size_t num_dims,
              const int64_t* src_dims, const void* src, const int* permutation,
              void* dst) override {
    SimplifyThenLaunch(stream_ctx, data_type, num_dims, src_dims, src, permutation, dst);
  }
};

class PermuteFactoryImpl : public PermuteFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PermuteFactoryImpl);
  PermuteFactoryImpl() = default;
  ~PermuteFactoryImpl() override = default;

  std::unique_ptr<Permute> New(size_t max_num_dims) override {
    if (max_num_dims <= kMaxNumDims) {
      return std::unique_ptr<Permute>(new PermuteImpl());
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, PermuteFactory, PermuteFactoryImpl);

}  // namespace

}  // namespace internal

}  // namespace permute

}  // namespace primitive

}  // namespace oneflow
