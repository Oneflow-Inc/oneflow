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
#ifndef ONEFLOW_USER_KERNELS_QUANTIZATION_UTILS_H_
#define ONEFLOW_USER_KERNELS_QUANTIZATION_UTILS_H_

#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/kernel/util/numeric_limits.cuh"

namespace oneflow {
namespace quantization {

constexpr int kWarpSize = 32;

template<int M>
__host__ __device__ __forceinline__ int ModDiv(int64_t N) {
  return N - (N / M * M);
}

template<>
__host__ __device__ __forceinline__ int ModDiv<2>(int64_t N) {
  return N & 0x1;
}

template<>
__host__ __device__ __forceinline__ int ModDiv<4>(int64_t N) {
  return N & 0x3;
}

template<>
__host__ __device__ __forceinline__ int ModDiv<8>(int64_t N) {
  return N & 0x7;
}

template<>
__host__ __device__ __forceinline__ int ModDiv<16>(int64_t N) {
  return N & 0xF;
}

template<int pack_size, typename T>
__global__ void ReduceMinMaxPerTensor(const int64_t elements, const T* in_ptr, T* min_max_ptr) {
  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  using MinMaxPack = cuda::elementwise::Pack<T, 2>;

  __shared__ T shared_buffer[kWarpSize << 1];
  MinMaxPack* shared_min_max = reinterpret_cast<MinMaxPack*>(shared_buffer);

  MinMaxPack min_max;
  min_max.elem[0] = detail::numeric_limits<T>::max();
  min_max.elem[1] = detail::numeric_limits<T>::lowest();

  int64_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x * pack_size;

  for (int64_t idx = tid * pack_size; idx < elements; idx += step) {
    LoadPack in;
    in.storage = reinterpret_cast<const LoadType*>(in_ptr + idx)[0];
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[i]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[i]);
    }
  }
  int rest = ModDiv<pack_size>(elements);
  if (rest > 0 && tid == (gridDim.x * blockDim.x - 1)) {
    in_ptr += elements - rest;
    LoadPack in;
    in.storage = reinterpret_cast<const LoadType*>(in_ptr)[0];
#pragma unroll
    for (int i = 0; i < rest; ++i) {
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[i]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[i]);
    }
  }

  for (int mask = kWarpSize >> 1; mask > 0; mask = mask >> 1) {
    T b_min = __shfl_down_sync(0xffffffff, min_max.elem[0], mask, kWarpSize);
    T b_max = __shfl_down_sync(0xffffffff, min_max.elem[1], mask, kWarpSize);
    min_max.elem[0] = BinaryFuncMin<T>::Invoke(b_min, min_max.elem[0]);
    min_max.elem[1] = BinaryFuncMax<T>::Invoke(b_max, min_max.elem[1]);
  }
  __syncthreads();

  // const int lid = threadIdx.x % kWarpSize;
  // const int wid = threadIdx.x / kWarpSize;
  // kWarpSize is 32
  const int lid = threadIdx.x & 0x1F;
  const int wid = threadIdx.x >> 5;

  if (lid == 0) { shared_min_max[wid] = min_max; }
  __syncthreads();

  if (wid == 0) {
    if (threadIdx.x < blockDim.x >> 5) {
      min_max = shared_min_max[lid];
    } else {
      min_max.elem[0] = detail::numeric_limits<T>::max();
      min_max.elem[1] = detail::numeric_limits<T>::lowest();
    }
    __syncwarp();

    for (int mask = kWarpSize >> 1; mask > 0; mask = mask >> 1) {
      T b_min = __shfl_down_sync(0xffffffff, min_max.elem[0], mask, kWarpSize);
      T b_max = __shfl_down_sync(0xffffffff, min_max.elem[1], mask, kWarpSize);
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(b_min, min_max.elem[0]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(b_max, min_max.elem[1]);
    }
    if (lid == 0) {
      reinterpret_cast<MinMaxPack*>(min_max_ptr)[blockIdx.x].storage = min_max.storage;
    }
  }
}

template<typename T, typename Q>
__global__ void ComputeScaleAndZeroPointBlock(const int min_max_size, const T* min_max_ptr,
                                              const Q upper_bound, const Q lower_bound,
                                              float* scale_ptr, Q* zero_point_ptr) {
  using MinMaxPack = cuda::elementwise::Pack<T, 2>;

  __shared__ T shared_buffer[kWarpSize << 1];
  MinMaxPack* shared_min_max = reinterpret_cast<MinMaxPack*>(shared_buffer);

  MinMaxPack min_max;
  min_max.elem[0] = detail::numeric_limits<T>::max();
  min_max.elem[1] = detail::numeric_limits<T>::lowest();
#pragma unroll
  for (int64_t idx = threadIdx.x; idx < min_max_size; idx += blockDim.x) {
    MinMaxPack in = reinterpret_cast<const MinMaxPack*>(min_max_ptr)[idx];
    min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[0]);
    min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[1]);
  }

  for (int mask = kWarpSize >> 1; mask > 0; mask = mask >> 1) {
    T b_min = __shfl_down_sync(0xffffffff, min_max.elem[0], mask, kWarpSize);
    T b_max = __shfl_down_sync(0xffffffff, min_max.elem[1], mask, kWarpSize);
    min_max.elem[0] = BinaryFuncMin<T>::Invoke(b_min, min_max.elem[0]);
    min_max.elem[1] = BinaryFuncMax<T>::Invoke(b_max, min_max.elem[1]);
  }
  __syncthreads();

  // const int lid = threadIdx.x % kWarpSize;
  // const int wid = threadIdx.x / kWarpSize;
  // kWarpSize is 32
  const int lid = threadIdx.x & 0x1F;
  const int wid = threadIdx.x >> 5;

  if (lid == 0) { shared_min_max[wid] = min_max; }
  __syncthreads();

  if (wid == 0) {
    if (threadIdx.x < blockDim.x >> 5) {
      min_max = shared_min_max[lid];
    } else {
      min_max.elem[0] = detail::numeric_limits<T>::max();
      min_max.elem[1] = detail::numeric_limits<T>::lowest();
    }
    __syncwarp();

    for (int mask = kWarpSize >> 1; mask > 0; mask = mask >> 1) {
      T b_min = __shfl_down_sync(0xffffffff, min_max.elem[0], mask, kWarpSize);
      T b_max = __shfl_down_sync(0xffffffff, min_max.elem[1], mask, kWarpSize);
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(b_min, min_max.elem[0]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(b_max, min_max.elem[1]);
    }
    if (lid == 0) {
      float min_value = static_cast<float>(min_max.elem[0]);
      float max_value = static_cast<float>(min_max.elem[1]);
      float scale = (max_value - min_value) / (upper_bound - lower_bound);
      int32_t zero_point = lower_bound - __float2int_rn(min_value / scale);
      scale_ptr[0] = scale;
      zero_point_ptr[0] = static_cast<Q>(zero_point);
    }
  }
}

template<>
inline __global__ void ComputeScaleAndZeroPointBlock<half, int8_t>(
    const int min_max_size, const half* min_max_ptr, const int8_t upper_bound,
    const int8_t lower_bound, float* scale_ptr, int8_t* zero_point_ptr) {
  using T = half;
  using Q = int8_t;
  using MinMaxPack4 = cuda::elementwise::Pack<T, 8>;
  using MinMaxPack = cuda::elementwise::Pack<T, 2>;

  __shared__ T shared_buffer[kWarpSize << 1];
  MinMaxPack* shared_min_max = reinterpret_cast<MinMaxPack*>(shared_buffer);

  MinMaxPack min_max;
  min_max.elem[0] = detail::numeric_limits<T>::max();
  min_max.elem[1] = detail::numeric_limits<T>::lowest();

#pragma unroll
  for (int idx = threadIdx.x; idx < (min_max_size >> 2); idx += blockDim.x) {
    MinMaxPack4 in = reinterpret_cast<const MinMaxPack4*>(min_max_ptr + (idx << 3))[0];
    min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[0]);
    min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[1]);
    min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[2]);
    min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[3]);
    min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[4]);
    min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[5]);
    min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[6]);
    min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[7]);
  }

  int rest = ModDiv<4>(min_max_size);

  if (rest > 0 && threadIdx.x == blockDim.x - 1) {
    int offset = (min_max_size - rest) << 1;
    MinMaxPack4 in = reinterpret_cast<const MinMaxPack4*>(min_max_ptr + offset)[0];
#pragma unroll
    for (int i = 0; i < rest; ++i) {
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[i << 1]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[(i << 1) + 1]);
    }
  }
  for (int mask = kWarpSize >> 1; mask > 0; mask = mask >> 1) {
    T b_min = __shfl_down_sync(0xffffffff, min_max.elem[0], mask, kWarpSize);
    T b_max = __shfl_down_sync(0xffffffff, min_max.elem[1], mask, kWarpSize);
    min_max.elem[0] = BinaryFuncMin<T>::Invoke(b_min, min_max.elem[0]);
    min_max.elem[1] = BinaryFuncMax<T>::Invoke(b_max, min_max.elem[1]);
  }
  __syncthreads();

  // const int lid = threadIdx.x % kWarpSize;
  // const int wid = threadIdx.x / kWarpSize;
  // kWarpSize is 32
  const int lid = threadIdx.x & 0x1F;
  const int wid = threadIdx.x >> 5;

  if (lid == 0) { shared_min_max[wid] = min_max; }
  __syncthreads();

  if (wid == 0) {
    if (threadIdx.x < blockDim.x >> 5) {
      min_max = shared_min_max[lid];
    } else {
      min_max.elem[0] = detail::numeric_limits<T>::max();
      min_max.elem[1] = detail::numeric_limits<T>::lowest();
    }
    __syncwarp();

    for (int mask = kWarpSize >> 1; mask > 0; mask = mask >> 1) {
      T b_min = __shfl_down_sync(0xffffffff, min_max.elem[0], mask, kWarpSize);
      T b_max = __shfl_down_sync(0xffffffff, min_max.elem[1], mask, kWarpSize);
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(b_min, min_max.elem[0]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(b_max, min_max.elem[1]);
    }
    if (lid == 0) {
      float min_value = static_cast<float>(min_max.elem[0]);
      float max_value = static_cast<float>(min_max.elem[1]);
      float scale = (max_value - min_value) / (upper_bound - lower_bound);
      int32_t zero_point = lower_bound - __float2int_rn(min_value / scale);
      scale_ptr[0] = scale;
      zero_point_ptr[0] = static_cast<Q>(zero_point);
    }
  }
}

template<int pack_size, typename T, typename Q>
__global__ void ApplyQuantization(const int64_t elements, const T* in_ptr, const float* scale_ptr,
                                  const Q* zero_point_ptr, const Q upper_bound, const Q lower_bound,
                                  Q* out_ptr) {
  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  using StoreType = cuda::elementwise::PackType<Q, pack_size>;
  using StorePack = cuda::elementwise::Pack<Q, pack_size>;

  int64_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x * pack_size;

  float scale = 1.f / *scale_ptr;
  float zero_point = *zero_point_ptr;

  for (int64_t idx = tid * pack_size; idx < elements; idx += step) {
    StorePack out;
    LoadPack in;
    in.storage = reinterpret_cast<const LoadType*>(in_ptr + idx)[0];
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      out.elem[i] =
          max(min(__float2int_rn(static_cast<float>(in.elem[i]) * scale + zero_point), upper_bound),
              lower_bound);
    }
    reinterpret_cast<StoreType*>(out_ptr + idx)[0] = out.storage;
  }

  int rest = ModDiv<pack_size>(elements);

  if (rest > 0 && tid == (gridDim.x * blockDim.x - 1)) {
    in_ptr += elements - rest;
    out_ptr += elements - rest;
    LoadPack in;
    in.storage = reinterpret_cast<const LoadType*>(in_ptr)[0];
#pragma unroll
    for (int i = 0; i < rest; ++i) {
      out_ptr[i] =
          max(min(__float2int_rn(static_cast<float>(in.elem[i]) * scale + zero_point), upper_bound),
              lower_bound);
    }
  }
}

template<typename T, typename Q>
inline void ApplyDynamicQuantization(cudaStream_t stream, const int min_max_size,
                                     const T* min_max_ptr, const int64_t elements, const T* in_ptr,
                                     const int quantization_bit, Q* out_ptr, float* scale_ptr,
                                     Q* zero_point_ptr) {
  Q upper_bound = (1 << (quantization_bit - 1)) - 1;
  Q lower_bound = -upper_bound - 1;
  size_t element_bytes = GetSizeOfDataType(GetDataType<T>::value);

  ComputeScaleAndZeroPointBlock<T, Q><<<1, cuda::elementwise::kBlockSize, 0, stream>>>(
      min_max_size, min_max_ptr, upper_bound, lower_bound, scale_ptr, zero_point_ptr);

  constexpr int pack_size = cuda::elementwise::PackSize<T>();
  int64_t pack_num = (elements + pack_size - 1) / pack_size;
  int grid_size = 0;
  cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  ApplyQuantization<pack_size, T, Q><<<grid_size, cuda::elementwise::kBlockSize, 0, stream>>>(
      elements, in_ptr, scale_ptr, zero_point_ptr, upper_bound, lower_bound, out_ptr);
}

}  // namespace quantization
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_QUANTIZATION_UTILS_H_
