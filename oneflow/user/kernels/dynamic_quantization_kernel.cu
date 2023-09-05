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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/kernel/util/numeric_limits.cuh"

namespace oneflow {

namespace {

template<int M>
__host__ __device__ int ModDiv(int64_t N) {
  return N - (N / M * M);
}

template<>
__host__ __device__ int ModDiv<2>(int64_t N) {
  return N & 0x1;
}

template<>
__host__ __device__ int ModDiv<4>(int64_t N) {
  return N & 0x3;
}

template<>
__host__ __device__ int ModDiv<8>(int64_t N) {
  return N & 0x7;
}

template<>
__host__ __device__ int ModDiv<16>(int64_t N) {
  return N & 0xF;
}

template<int pack_size, typename T>
__global__ void ReduceMinMaxPerTensor(const int64_t elements, const T* in_ptr, T* min_max_ptr) {
  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  using MinMaxPack = cuda::elementwise::Pack<T, 2>;

  extern __shared__ uint8_t buffer[];

  MinMaxPack min_max;
  min_max.elem[0] = detail::numeric_limits<T>::max();
  min_max.elem[1] = detail::numeric_limits<T>::lowest();

  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x * pack_size;

  for (int64_t idx = gid * pack_size; idx < elements; idx += step) {
    LoadPack in;
    in.storage = reinterpret_cast<const LoadType*>(in_ptr + idx)[0];
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[i]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[i]);
    }
  }
  int rest = ModDiv<pack_size>(elements);
  if (rest > 0 && gid == (gridDim.x * blockDim.x - 1)) {
    in_ptr += elements - rest;
    LoadPack in;
    in.storage = reinterpret_cast<const LoadType*>(in_ptr)[0];
#pragma unroll
    for (int i = 0; i < rest; ++i) {
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[i]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[i]);
    }
  }

  int64_t tid = threadIdx.x;

  MinMaxPack* shared_min_max = reinterpret_cast<MinMaxPack*>(buffer);
  shared_min_max[tid].storage = min_max.storage;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      MinMaxPack min_max0, min_max1;
      min_max0.storage = shared_min_max[tid].storage;
      min_max1.storage = shared_min_max[tid + s].storage;
      min_max0.elem[0] = BinaryFuncMin<T>::Invoke(min_max0.elem[0], min_max1.elem[0]);
      min_max0.elem[1] = BinaryFuncMax<T>::Invoke(min_max0.elem[1], min_max1.elem[1]);
      shared_min_max[tid].storage = min_max0.storage;
    }
    __syncthreads();
  }

  if (tid == 0) {
    reinterpret_cast<MinMaxPack*>(min_max_ptr)[blockIdx.x].storage = shared_min_max[0].storage;
  }
}

template<typename T, typename Q>
__global__ void ComputeScaleAndZeroPointBlock(const int min_max_size, const T* min_max_ptr,
                                              const Q upper_bound, const Q lower_bound,
                                              float* scale_ptr, Q* zero_point_ptr) {
  using MinMaxPack = cuda::elementwise::Pack<T, 2>;

  extern __shared__ uint8_t buffer[];
  MinMaxPack* shared_min_max = reinterpret_cast<MinMaxPack*>(buffer);
  int64_t tid = threadIdx.x;
  {
    MinMaxPack min_max;
    min_max.elem[0] = detail::numeric_limits<T>::max();
    min_max.elem[1] = detail::numeric_limits<T>::lowest();
#pragma unroll
    for (int64_t idx = tid; idx < min_max_size; idx += blockDim.x) {
      MinMaxPack in = reinterpret_cast<const MinMaxPack*>(min_max_ptr)[idx];
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[0]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[1]);
    }
    shared_min_max[tid].storage = min_max.storage;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        MinMaxPack min_max0, min_max1;
        min_max0.storage = shared_min_max[tid].storage;
        min_max1.storage = shared_min_max[tid + s].storage;
        min_max0.elem[0] = BinaryFuncMin<T>::Invoke(min_max0.elem[0], min_max1.elem[0]);
        min_max0.elem[1] = BinaryFuncMax<T>::Invoke(min_max0.elem[1], min_max1.elem[1]);
        shared_min_max[tid].storage = min_max0.storage;
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    MinMaxPack min_max = shared_min_max[0];
    float min_value = static_cast<float>(min_max.elem[0]);
    float max_value = static_cast<float>(min_max.elem[1]);
    float scale = (max_value - min_value) / (upper_bound - lower_bound);
    int32_t zero_point = lower_bound - __float2int_rn(min_value / scale);
    scale_ptr[0] = scale;
    zero_point_ptr[0] = static_cast<Q>(zero_point);
  }
}

template<>
__global__ void ComputeScaleAndZeroPointBlock<half, int8_t>(
    const int min_max_size, const half* min_max_ptr, const int8_t upper_bound,
    const int8_t lower_bound, float* scale_ptr, int8_t* zero_point_ptr) {
  using T = half;
  using Q = int8_t;
  using MinMaxPack4 = cuda::elementwise::Pack<T, 8>;
  using MinMaxPack = cuda::elementwise::Pack<T, 2>;

  extern __shared__ uint8_t buffer[];
  MinMaxPack* shared_min_max = reinterpret_cast<MinMaxPack*>(buffer);
  int64_t tid = threadIdx.x;

  MinMaxPack min_max;
  min_max.elem[0] = detail::numeric_limits<T>::max();
  min_max.elem[1] = detail::numeric_limits<T>::lowest();

#pragma unroll
  for (int idx = tid; idx < (min_max_size >> 2); idx += blockDim.x) {
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

  if (rest > 0 && tid == blockDim.x - 1) {
    int offset = (min_max_size - rest) << 1;
    MinMaxPack4 in = reinterpret_cast<const MinMaxPack4*>(min_max_ptr + offset)[0];
#pragma unroll
    for (int i = 0; i < rest; ++i) {
      min_max.elem[0] = BinaryFuncMin<T>::Invoke(min_max.elem[0], in.elem[i << 1]);
      min_max.elem[1] = BinaryFuncMax<T>::Invoke(min_max.elem[1], in.elem[(i << 1) + 1]);
    }
  }

  shared_min_max[tid].storage = min_max.storage;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      MinMaxPack min_max0, min_max1;
      min_max0.storage = shared_min_max[tid].storage;
      min_max1.storage = shared_min_max[tid + s].storage;
      min_max0.elem[0] = BinaryFuncMin<T>::Invoke(min_max0.elem[0], min_max1.elem[0]);
      min_max0.elem[1] = BinaryFuncMax<T>::Invoke(min_max0.elem[1], min_max1.elem[1]);
      shared_min_max[tid].storage = min_max0.storage;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    MinMaxPack min_max = shared_min_max[0];
    float min_value = static_cast<float>(min_max.elem[0]);
    float max_value = static_cast<float>(min_max.elem[1]);
    float scale = (max_value - min_value) / (upper_bound - lower_bound);
    int32_t zero_point = lower_bound - __float2int_rn(min_value / scale);
    scale_ptr[0] = scale;
    zero_point_ptr[0] = static_cast<Q>(zero_point);
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

  float scale = *scale_ptr;
  float zero_point = *zero_point_ptr;

  for (int64_t idx = tid * pack_size; idx < elements; idx += step) {
    StorePack out;
    LoadPack in;
    in.storage = reinterpret_cast<const LoadType*>(in_ptr + idx)[0];
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      out.elem[i] =
          max(min(__float2int_rn(static_cast<float>(in.elem[i]) / scale + zero_point), upper_bound),
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
          max(min(__float2int_rn(static_cast<float>(in.elem[i]) / scale + zero_point), upper_bound),
              lower_bound);
    }
  }
}

template<typename T, typename Q>
void ApplyDynamicQuantization(cudaStream_t stream, const int min_max_size, const T* min_max_ptr,
                              const int64_t elements, const T* in_ptr, const int quantization_bit,
                              Q* out_ptr, float* scale_ptr, Q* zero_point_ptr) {
  Q upper_bound = (1 << (quantization_bit - 1)) - 1;
  Q lower_bound = -upper_bound - 1;
  size_t element_bytes = GetSizeOfDataType(GetDataType<T>::value);

  ComputeScaleAndZeroPointBlock<T, Q>
      <<<1, cuda::elementwise::kBlockSize, cuda::elementwise::kBlockSize * element_bytes * 2,
         stream>>>(min_max_size, min_max_ptr, upper_bound, lower_bound, scale_ptr, zero_point_ptr);

  constexpr int pack_size = cuda::elementwise::PackSize<T>();
  int64_t pack_num = (elements + pack_size - 1) / pack_size;
  int grid_size = 0;
  cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  ApplyQuantization<pack_size, T, Q><<<grid_size, cuda::elementwise::kBlockSize, 0, stream>>>(
      elements, in_ptr, scale_ptr, zero_point_ptr, upper_bound, lower_bound, out_ptr);
}

}  // namespace

template<typename T>
class GpuDynamicQuantizationKernel final : public user_op::OpKernel {
 public:
  GpuDynamicQuantizationKernel() = default;
  ~GpuDynamicQuantizationKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor* zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const bool per_layer_quantization = ctx->Attr<bool>("per_layer_quantization");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    CHECK(quantization_scheme == "affine");

    const int64_t elements = in->shape_view().elem_cnt();

    constexpr int pack_size = cuda::elementwise::PackSize<T>();
    int64_t pack_num = (elements + pack_size - 1) / pack_size;
    int grid_size = 0;
    cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
    grid_size = grid_size > 2048 ? 2048 : grid_size;

    size_t element_bytes = GetSizeOfDataType(GetDataType<T>::value);
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), grid_size * element_bytes * 2);

    T* min_max = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    auto stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    if (per_layer_quantization) {
      ReduceMinMaxPerTensor<pack_size, T>
          <<<grid_size, cuda::elementwise::kBlockSize,
             cuda::elementwise::kBlockSize * element_bytes * 2, stream>>>(elements, in->dptr<T>(),
                                                                          min_max);
    } else {
      UNIMPLEMENTED() << "dynamic_quantization does not support per-channel quantization";
    }

    if (quantization_formula == "oneflow") {
      if (quantization_bit == 8) {
        ApplyDynamicQuantization<T, int8_t>(
            stream, grid_size, min_max, elements, in->dptr<T>(), quantization_bit,
            out->mut_dptr<int8_t>(), scale->mut_dptr<float>(), zero_point->mut_dptr<int8_t>());
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED() << "dynamic_quantization only support oneflow quantization formula";
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DYNAMIC_QUANTIZATION_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("dynamic_quantization")                                          \
      .SetCreateFn<GpuDynamicQuantizationKernel<dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t { return 128 * 1024 * 1024; })

REGISTER_DYNAMIC_QUANTIZATION_KERNEL(float);
REGISTER_DYNAMIC_QUANTIZATION_KERNEL(double);
REGISTER_DYNAMIC_QUANTIZATION_KERNEL(half);

}  // namespace oneflow
