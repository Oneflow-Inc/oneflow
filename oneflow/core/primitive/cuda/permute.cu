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

constexpr int32_t TILE_SIZE = 32; // float tile size. 
constexpr int32_t HALF_TILE_SIZE = 64; // cause float16 is half of float32, we need to double tilesize for half kernel. 
constexpr int32_t KBLOCKROWS = 8;

// (B, X, Y) -> (B, Y, X), refer from
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template<size_t num_dims, size_t movement_size, size_t tile_size, typename IndexType>
__global__ void BatchPermuteKernel(PermuteKernelParams<num_dims, IndexType> params, IndexType H,
                                   IndexType W, IndexType dh, IndexType dw, int32_t grid_size) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  __shared__ T tile[tile_size][tile_size + 1];  // To avoid bank conflict.
  const T* src = reinterpret_cast<const T*>(params.src);
  T* dst = reinterpret_cast<T*>(params.dst);
  IndexType dh_mul_dw = dh * dw; 
  for (int i = blockIdx.x, step = gridDim.x; i < grid_size; i += step) {
    const IndexType n = i / dh_mul_dw;  // the index of batch.
    const IndexType k = i - n * dh_mul_dw;  // equal to i % (dh*dw). the flatten index of tile in a batch. 

    const IndexType r = k / dw;  // the row index of tile in a batch.
    const IndexType c = k - r * dw; // equal to k % dw. the col index of tile in a batch.
    const IndexType offset = n * H * W;
    int x = c * tile_size + threadIdx.x;
    int y = r * tile_size + threadIdx.y;
    if (x < W) {
#pragma unroll
      // each thread process 4 elements.
      for (int i = 0; threadIdx.y + i < tile_size && y + i < H; i += KBLOCKROWS) {
        tile[threadIdx.y + i][threadIdx.x] = src[offset + (y + i) * W + x];
      }
    }
    __syncthreads();
    x = r * tile_size + threadIdx.x;
    y = c * tile_size + threadIdx.y;
    if (x < H) {
#pragma unroll
      for (int i = 0; threadIdx.y + i < tile_size && y + i < W; i += KBLOCKROWS) {
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
template<size_t num_dims, size_t movement_size, size_t tile_size, typename IndexType>
__global__ void BatchPermuteHalf2Kernel(PermuteKernelParams<num_dims, IndexType> params, IndexType H,
                                   IndexType W, IndexType dh, IndexType dw, int32_t grid_size) {
  // Use union structure to process Load and Store. 
  __shared__ union {
    half tile_half[tile_size][tile_size+2]; // [64][66]
    half2 tile_half2[tile_size/2][tile_size/2+1]; // [32][33]
  } tile_mem;

  const half2* src = reinterpret_cast<const half2*>(params.src);
  half* dst = reinterpret_cast<half*>(params.dst);
  
  IndexType dh_mul_dw = dh * dw; 
  for (int i = blockIdx.x, step = gridDim.x; i < grid_size; i += step) {
    const IndexType n = i / dh_mul_dw;  // the index of batch.
    const IndexType k = i - n * dh_mul_dw;  // equal to i%(dh*dw). the flatten index of tile in a batch. TODO! optimize it!

    const IndexType r = k / dw;  // the row index of tile in a batch.
    const IndexType c = k - r * dw; // equal to k % dw. the col index of tile in a batch.
    const IndexType offset = n * H * W;
    int x = c * tile_size + threadIdx.x*2; // cause each thread process a half2 element, we need to multiply 2 for threadIdx.x. 
    int y = r * tile_size + threadIdx.y;
    if (x < W) {
#pragma unroll
      // each thread process 4 elements.
      for (int i = 0; threadIdx.y + i < tile_size && y + i < H; i += KBLOCKROWS) {
        // each thread load a half2. 
        tile_mem.tile_half2[threadIdx.y + i][threadIdx.x] = src[(offset + (y + i) * W + x)/2];
      }
    }
    __syncthreads();
    x = r * tile_size + threadIdx.x*2; // cause each thread process a half2 element, we need to multiply 2 for threadIdx.x. 
    y = c * tile_size + threadIdx.y; 
    if (x < H) {
#pragma unroll
      for (int i = 0; threadIdx.y + i < tile_size && y + i < W; i += KBLOCKROWS) {
        /*
        When write back as column, it cannot be stored as half2 directly. 
        So we split as 2 half elements, and write back separately. 
        */
        dst[offset + (y + i) * H + x] = tile_mem.tile_half[threadIdx.x*2][threadIdx.y + i]; 
        dst[offset + (y + i) * H + x + 1] = tile_mem.tile_half[threadIdx.x*2 + 1][threadIdx.y + i]; 
      }
    }
    __syncthreads();
  }
}


template<size_t num_dims, size_t movement_size, size_t tile_size, typename IndexType>
void LaunchBatchPermuteKernel(StreamContext* stream_ctx,
                              PermuteKernelParams<num_dims, IndexType> params, IndexType& n,
                              IndexType& h, IndexType& w) {
  cudaStream_t cuda_stream =
      CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();

  IndexType dh = (h + tile_size - 1) / tile_size;
  IndexType dw = (w + tile_size - 1) / tile_size;

  const int32_t grid_size = n * dh * dw;
  int32_t checked_grid_size = std::min(grid_size, kCudaMaxBlocksNum); 
  if(tile_size == HALF_TILE_SIZE){
    const int32_t half2_thread = tile_size / 2; // cause each thread process two half elements. 
    // Only specialized movementsize==2 to avoid using too much shared memory. 
    BatchPermuteHalf2Kernel<num_dims, 2, HALF_TILE_SIZE, IndexType>
      <<<checked_grid_size, dim3(half2_thread, KBLOCKROWS), 0, cuda_stream>>>(
          params, h, w, dh, dw, grid_size);  // Set threads num as 32x8 cause each threads process 4
                                             // elements to 32x32 share memory.
  }else{
    BatchPermuteKernel<num_dims, movement_size, tile_size, IndexType>
      <<<checked_grid_size, dim3(tile_size, KBLOCKROWS), 0, cuda_stream>>>(
          params, h, w, dh, dw, grid_size);  
  }
}


template<size_t tile_size, typename IndexType>
bool CheckIfGreaterThanTileSize(IndexType& h, IndexType& w) {
  // H W should be less than tile size.
  if (h < tile_size || w < tile_size) { return false; }
  return true;
}

template<size_t num_dims, size_t tile_size, typename IndexType>
bool CheckLaunchBatchPermute(PermuteKernelParams<num_dims, IndexType> params, IndexType& n,
                             IndexType& h, IndexType& w) {
  if (CheckIfGreaterThanTileSize<tile_size, IndexType>(h, w)) {
    if (n == 1) {
      return true;
    } else if (num_dims == 3 && params.permutation[2] == 1 && params.permutation[1] == 2) {
      return true;
    } else {
      return false;
    }
  }
  return false;
}

template<typename IndexType, size_t movement_size>
bool CheckUseHalf2(IndexType& h, IndexType& w) {
  if(movement_size == 2){ // movement_size = 2, means half type
    if(h % 2 == 0 && w % 2 ==0){
      // When h and w can both divided by 2, it means we can use half2 type to improve load efficiency. 
      return true; 
    }
  }
  return false;
}


template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(StreamContext* stream_ctx, PermuteKernelParams<num_dims, IndexType> params) {
  cudaStream_t cuda_stream =
      CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx))->cuda_stream();

  if (num_dims == 2 || num_dims == 3) {
    IndexType n;
    IndexType h;
    IndexType w;
    if (num_dims == 2) {
      IndexType global_index[2];
      /*
      For example: assume dim is (4, 6), offset = 24.
      offset-1=23, convet back to NdIndex is (3, 5), cause index start from zero and we need to
      subtract 1. then we add 1 to all the NdIndex to get the actual dim.
      */
      params.src_index_helper.OffsetToNdIndex(params.count - 1, global_index);
      n = 1;
      h = global_index[0] + 1;
      w = global_index[1] + 1;
    } else {
      IndexType global_index[3];
      params.src_index_helper.OffsetToNdIndex(params.count - 1, global_index);
      n = global_index[0] + 1;
      h = global_index[1] + 1;
      w = global_index[2] + 1;
    }
    if (CheckLaunchBatchPermute<num_dims, TILE_SIZE>(params, n, h, w)) {
      if(CheckUseHalf2<IndexType, movement_size>(h, w)){
        LaunchBatchPermuteKernel<num_dims, 2, HALF_TILE_SIZE, IndexType>(stream_ctx, params, n, h, w);  
      }else{
        LaunchBatchPermuteKernel<num_dims, movement_size, TILE_SIZE, IndexType>(stream_ctx, params, n, h, w);    
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
