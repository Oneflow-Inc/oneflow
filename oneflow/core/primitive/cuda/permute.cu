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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/primitive/include/permute.h"
#include "oneflow/core/primitive/common/permute.h"
#include "oneflow/core/stream/cuda_stream_context.h"
#include "oneflow/core/primitive/cuda/cuda_graph_support.h"
#include <cuda_runtime.h>

namespace oneflow {

namespace primitive {

namespace permute_internal {

namespace {

constexpr int32_t kTileSize = 32;  // float tile size.
constexpr int32_t kMov2TileSize =
    64;  // cause float16 is half of float32, we need to double tilesize for half kernel.
constexpr int32_t kBlockRows = 8;

// Naive Version.
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
__global__ void BatchPermuteKernel(const void* src_ptr, void* dst_ptr, IndexType H, IndexType W,
                                   IndexType num_tile_rows, IndexType num_tile_cols,
                                   int32_t grid_size) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  __shared__ T tile[tile_size][tile_size + 1];  // To avoid bank conflict.

  const T* src = reinterpret_cast<const T*>(src_ptr);
  T* dst = reinterpret_cast<T*>(dst_ptr);

  IndexType batch_num_tile = num_tile_rows * num_tile_cols;
  for (int i = blockIdx.x, step = gridDim.x; i < grid_size; i += step) {
    const IndexType batch_index = i / batch_num_tile;  // the index of batch.
    const IndexType k =
        i - batch_index * batch_num_tile;  // equal to i % (num_tile_rows*num_tile_cols). the
                                           // flatten index of tile in a batch.

    const IndexType r = k / num_tile_cols;  // the row index of tile in a batch.
    const IndexType c =
        k - r * num_tile_cols;  // equal to k % num_tile_cols. the col index of tile in a batch.
    const IndexType offset = batch_index * H * W;
    IndexType x = c * tile_size + threadIdx.x;
    IndexType y = r * tile_size + threadIdx.y;
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
    x = r * tile_size + threadIdx.x;
    y = c * tile_size + threadIdx.y;
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
Here is a half2 version of Batch Permute.
When the H W can be divided by 2. we can read data use half2, and write back as half.
We design a union structure to store half and half2 share memory.
Actually here shared memory size is Half[64][66].
*/
template<size_t num_dims, size_t tile_size, typename IndexType>
__global__ void BatchPermuteMovement2Kernel(const void* src_ptr, void* dst_ptr, IndexType H,
                                            IndexType W, IndexType num_tile_rows,
                                            IndexType num_tile_cols, int32_t grid_size) {
  static_assert(tile_size % 2 == 0);
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
  for (int i = blockIdx.x, step = gridDim.x; i < grid_size; i += step) {
    const IndexType batch_index = i / batch_num_tile;  // the index of batch.
    const IndexType k =
        i - batch_index * batch_num_tile;  // equal to i%(num_tile_rows*num_tile_cols). the flatten
                                           // index of tile in a batch.

    const IndexType r = k / num_tile_cols;  // the row index of tile in a batch.
    const IndexType c =
        k - r * num_tile_cols;  // equal to k % num_tile_cols. the col index of tile in a batch.
    const IndexType offset = batch_index * H * W;
    int x = c * tile_size + threadIdx.x * 2;  // cause each thread process a half2 element, we need
                                              // to multiply 2 for threadIdx.x.
    int y = r * tile_size + threadIdx.y;
    if (x < W) {
      // each thread process 4 elements.
      IndexType y_range =
          ((tile_size - threadIdx.y) < (H - y)) ? (tile_size - threadIdx.y) : (H - y);
#pragma unroll
      // `i < y_range` equals to: `threadIdx.y + i < tile_size && y + i < H`.
      for (int i = 0; i < y_range; i += kBlockRows) {
        // each thread load a half2.
        tile_mem.tile_m4[threadIdx.y + i][threadIdx.x] = src[(offset + (y + i) * W + x) / 2];
      }
    }
    __syncthreads();
    x = r * tile_size + threadIdx.x * 2;  // cause each thread process a half2 element, we need to
                                          // multiply 2 for threadIdx.x.
    y = c * tile_size + threadIdx.y;
    if (x < H) {
      IndexType x_range =
          ((tile_size - threadIdx.y) < (W - y)) ? (tile_size - threadIdx.y) : (W - y);
#pragma unroll
      // `i < x_range` equals to: `threadIdx.y + i < tile_size && y + i < W`.
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
        dst[(offset + (y + i) * H + x) / 2] = tmp_storage.m4;
      }
    }
    __syncthreads();
  }
}

template<size_t num_dims, size_t movement_size, size_t tile_size, typename IndexType>
void LaunchBatchPermuteKernel(cudaStream_t& cuda_stream,
                              const PermuteKernelParams<num_dims, IndexType>& params,
                              const IndexType& n, const IndexType& h, const IndexType& w) {
  IndexType num_tile_rows = (h + tile_size - 1) / tile_size;
  IndexType num_tile_cols = (w + tile_size - 1) / tile_size;

  const int32_t grid_size = n * num_tile_rows * num_tile_cols;
  int32_t checked_grid_size = std::min(grid_size, kCudaMaxBlocksNum);
  if (tile_size == kMov2TileSize) {
    const int32_t half2_thread = tile_size / 2;  // cause each thread process two half elements.
    BatchPermuteMovement2Kernel<num_dims, kMov2TileSize, IndexType>
        <<<checked_grid_size, dim3(half2_thread, kBlockRows), 0, cuda_stream>>>(
            params.src, params.dst, h, w, num_tile_rows, num_tile_cols,
            grid_size);  // Set threads num as 32x8 cause each threads
                         // process 4 elements to 32x32 share memory.
  } else {
    BatchPermuteKernel<num_dims, movement_size, tile_size, IndexType>
        <<<checked_grid_size, dim3(tile_size, kBlockRows), 0, cuda_stream>>>(
            params.src, params.dst, h, w, num_tile_rows, num_tile_cols, grid_size);
  }
}

template<size_t tile_size, typename IndexType>
bool CheckIfGreaterEqualThanTileSize(const IndexType* h, const IndexType* w) {
  // H W should be less than tile size.
  if (*h < tile_size || *w < tile_size) { return false; }
  return true;
}

template<size_t num_dims, size_t tile_size, typename IndexType>
bool CheckLaunchBatchPermute(const int permutation[num_dims], const IndexType* n,
                             const IndexType* h, const IndexType* w) {
  if (CheckIfGreaterEqualThanTileSize<tile_size, IndexType>(h, w)) {
    if (*n == 1) {
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
bool CheckUseHalf2(IndexType* h, IndexType* w) {
  return movement_size == 2 && *h % 2 == 0 && *w % 2 == 0;
}

template<size_t num_dims, typename IndexType>
void InferBatchPermuteShape(const NdIndexOffsetHelper<IndexType, num_dims>* src_dims,
                            const IndexType* count, IndexType* num_batches, IndexType* rows,
                            IndexType* cols) {
  if (num_dims == 2) {
    IndexType global_index[2];
    /*
    For example: assume dim is (4, 6), offset = 24.
    offset-1=23, convet back to NdIndex is (3, 5), cause index start from zero and we need to
    subtract 1. then we add 1 to all the NdIndex to get the actual dim.
    */
    src_dims->OffsetToNdIndex(*count - 1, global_index);
    *num_batches = 1;
    *rows = global_index[0] + 1;
    *cols = global_index[1] + 1;
  } else {
    IndexType global_index[3];
    src_dims->OffsetToNdIndex(*count - 1, global_index);
    *num_batches = global_index[0] + 1;
    *rows = global_index[1] + 1;
    *cols = global_index[2] + 1;
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
    IndexType n;
    IndexType h;
    IndexType w;
    InferBatchPermuteShape<num_dims, IndexType>(&params.src_index_helper, &params.count, &n, &h,
                                                &w);
    if (CheckLaunchBatchPermute<num_dims, kTileSize>(params.permutation, &n, &h, &w)) {
      if (CheckUseHalf2<IndexType, movement_size>(&h, &w)) {
        LaunchBatchPermuteKernel<num_dims, 2, kMov2TileSize, IndexType>(cuda_stream, params, n, h,
                                                                        w);
      } else {
        LaunchBatchPermuteKernel<num_dims, movement_size, kTileSize, IndexType>(cuda_stream, params,
                                                                                n, h, w);
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

class PermuteImpl : public Permute, public CudaGraphSupport {
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

}  // namespace permute_internal

}  // namespace primitive

}  // namespace oneflow
