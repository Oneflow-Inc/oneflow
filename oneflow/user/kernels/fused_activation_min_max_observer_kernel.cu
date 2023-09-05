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
__global__ void ReduceMinMaxPerLayer(const int64_t elements, const T* in_ptr, T* min_max_ptr) {
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
__global__ void ComputeOFScaleAndZeroPoint(const T* min_max_ptr, const int min_max_size,
                                           const int quantization_bit, const float* weight_scale,
                                           const float* weight_acc, const T* bias, T* in_scale,
                                           Q* in_zero_point, T* out_scale, T* out_bias,
                                           const int out_elements) {
  using MinMaxPack = cuda::elementwise::Pack<T, 2>;

  extern __shared__ uint8_t buffer[];
  MinMaxPack* shared_min_max = reinterpret_cast<MinMaxPack*>(buffer);
  int64_t tid = threadIdx.x;
  {
    MinMaxPack min_max;
    min_max.elem[0] = detail::numeric_limits<T>::max();
    min_max.elem[1] = detail::numeric_limits<T>::lowest();
#pragma unroll
    for (int64_t idx = threadIdx.x; idx < min_max_size; idx += blockDim.x) {
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

  MinMaxPack min_max = shared_min_max[0];
  float min_value = static_cast<float>(min_max.elem[0]);
  float max_value = static_cast<float>(min_max.elem[1]);
  float input_scale = (max_value - min_value) / ((1 << quantization_bit) - 1);
  int32_t input_zero_point =
      -(__float2int_rn(min_value / input_scale) + (1 << (quantization_bit - 1)));
  float scale_zero_point = -input_scale * input_zero_point;

  int64_t thread_num = gridDim.x * blockDim.x;
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (gid == 0) {
    in_scale[0] = static_cast<T>(input_scale);
    in_zero_point[0] = static_cast<Q>(input_zero_point);
  }

  using LoadWPack = cuda::elementwise::Pack<float, 4>;
  using LoadBPack = cuda::elementwise::Pack<T, 4>;
  using StorePack = cuda::elementwise::Pack<T, 4>;

  if (bias) {
    for (int64_t idx = gid << 2; idx < out_elements; idx += thread_num << 2) {
      LoadWPack w_scale = reinterpret_cast<const LoadWPack*>(weight_scale + idx)[0];
      LoadWPack w_acc = reinterpret_cast<const LoadWPack*>(weight_acc + idx)[0];
      LoadBPack b = reinterpret_cast<const LoadBPack*>(bias + idx)[0];
      StorePack store_scale, store_bias;

      store_scale.elem[0] = static_cast<T>(w_scale.elem[0] * input_scale);
      store_scale.elem[1] = static_cast<T>(w_scale.elem[1] * input_scale);
      store_scale.elem[2] = static_cast<T>(w_scale.elem[2] * input_scale);
      store_scale.elem[3] = static_cast<T>(w_scale.elem[3] * input_scale);

      store_bias.elem[0] = static_cast<T>(__fmaf_rn(w_acc.elem[0], scale_zero_point, b.elem[0]));
      store_bias.elem[1] = static_cast<T>(__fmaf_rn(w_acc.elem[1], scale_zero_point, b.elem[1]));
      store_bias.elem[2] = static_cast<T>(__fmaf_rn(w_acc.elem[2], scale_zero_point, b.elem[2]));
      store_bias.elem[3] = static_cast<T>(__fmaf_rn(w_acc.elem[3], scale_zero_point, b.elem[3]));

      reinterpret_cast<StorePack*>(out_scale + idx)[0] = store_scale;
      reinterpret_cast<StorePack*>(out_bias + idx)[0] = store_bias;
    }
    int rest = ModDiv<4>(out_elements);
    if (rest > 0 && gid == (thread_num - 1)) {
      int offset = out_elements - rest;
      LoadWPack w_scale = reinterpret_cast<const LoadWPack*>(weight_scale + offset)[0];
      LoadWPack w_acc = reinterpret_cast<const LoadWPack*>(weight_acc + offset)[0];
      LoadBPack b = reinterpret_cast<const LoadBPack*>(bias + offset)[0];
#pragma unroll
      for (int i = 0; i < rest; ++i) {
        out_scale[offset + i] = static_cast<T>(w_scale.elem[i] * input_scale);
        out_bias[offset + i] =
            static_cast<T>(__fmaf_rn(w_acc.elem[i], scale_zero_point, b.elem[i]));
      }
    }
  } else {
    for (int64_t idx = gid << 2; idx < out_elements; idx += thread_num << 2) {
      LoadWPack w_scale = reinterpret_cast<const LoadWPack*>(weight_scale + idx)[0];
      LoadWPack w_acc = reinterpret_cast<const LoadWPack*>(weight_acc + idx)[0];
      StorePack store_scale, store_bias;

      store_scale.elem[0] = static_cast<T>(w_scale.elem[0] * input_scale);
      store_scale.elem[1] = static_cast<T>(w_scale.elem[1] * input_scale);
      store_scale.elem[2] = static_cast<T>(w_scale.elem[2] * input_scale);
      store_scale.elem[3] = static_cast<T>(w_scale.elem[3] * input_scale);

      store_bias.elem[0] = static_cast<T>(w_acc.elem[0] * scale_zero_point);
      store_bias.elem[1] = static_cast<T>(w_acc.elem[1] * scale_zero_point);
      store_bias.elem[2] = static_cast<T>(w_acc.elem[2] * scale_zero_point);
      store_bias.elem[3] = static_cast<T>(w_acc.elem[3] * scale_zero_point);

      reinterpret_cast<StorePack*>(out_scale + idx)[0] = store_scale;
      reinterpret_cast<StorePack*>(out_bias + idx)[0] = store_bias;
    }
    int rest = ModDiv<4>(out_elements);
    if (rest > 0 && gid == (thread_num - 1)) {
      int offset = out_elements - rest;
      LoadWPack w_scale = reinterpret_cast<const LoadWPack*>(weight_scale + offset)[0];
      LoadWPack w_acc = reinterpret_cast<const LoadWPack*>(weight_acc + offset)[0];
#pragma unroll
      for (int i = 0; i < rest; ++i) {
        out_scale[offset + i] = static_cast<T>(w_scale.elem[i] * input_scale);
        out_bias[offset + i] = static_cast<T>(w_acc.elem[i] * scale_zero_point);
      }
    }
  }
}

}  // namespace

template<typename T>
class GpuFusedActivationMinMaxObserverKernel final : public user_op::OpKernel {
 public:
  GpuFusedActivationMinMaxObserverKernel() = default;
  ~GpuFusedActivationMinMaxObserverKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight_scale = ctx->Tensor4ArgNameAndIndex("weight_scale", 0);
    const user_op::Tensor* weight_acc = ctx->Tensor4ArgNameAndIndex("weight_acc", 0);
    const user_op::Tensor* bias = nullptr;
    if (ctx->has_input("bias", 0)) { bias = ctx->Tensor4ArgNameAndIndex("bias", 0); }

    user_op::Tensor* in_scale = ctx->Tensor4ArgNameAndIndex("in_scale", 0);
    user_op::Tensor* in_zero_point = ctx->Tensor4ArgNameAndIndex("in_zero_point", 0);
    user_op::Tensor* out_scale = ctx->Tensor4ArgNameAndIndex("out_scale", 0);
    user_op::Tensor* out_bias = ctx->Tensor4ArgNameAndIndex("out_bias", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const bool per_layer_quantization = ctx->Attr<bool>("per_layer_quantization");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    CHECK(quantization_scheme == "affine");
    CHECK(quantization_bit == 8);

    const int64_t elements = in->shape_view().elem_cnt();

    constexpr int pack_size = cuda::elementwise::PackSize<T>();
    int grid_size = 0;
    int64_t pack_num = (elements + pack_size - 1) / pack_size;
    cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
    grid_size = grid_size > 2048 ? 2048 : grid_size;

    size_t element_bytes = GetSizeOfDataType(GetDataType<T>::value);
    CHECK_GE(tmp_buffer->shape_view().elem_cnt(), grid_size * element_bytes * 2);

    T* min_max = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    auto stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    if (per_layer_quantization) {
      ReduceMinMaxPerLayer<pack_size, T>
          <<<grid_size, cuda::elementwise::kBlockSize,
             cuda::elementwise::kBlockSize * element_bytes * 2, stream>>>(elements, in->dptr<T>(),
                                                                          min_max);
    } else {
      UNIMPLEMENTED()
          << "fused_activation_min_max_observer does not support per-channel quantization";
    }

    if (quantization_formula == "oneflow") {
      if (quantization_bit == 8) {
        ComputeOFScaleAndZeroPoint<T, int8_t>
            <<<1, cuda::elementwise::kBlockSize, cuda::elementwise::kBlockSize * element_bytes * 2,
               stream>>>(min_max, grid_size, quantization_bit, weight_scale->dptr<float>(),
                         weight_acc->dptr<float>(), bias ? bias->dptr<T>() : nullptr,
                         in_scale->mut_dptr<T>(), in_zero_point->mut_dptr<int8_t>(),
                         out_scale->mut_dptr<T>(), out_bias->mut_dptr<T>(),
                         out_scale->shape_view().elem_cnt());
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED()
          << "fused_activation_min_max_observer only support oneflow quantization formula";
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_ACTIVATION_MIN_MAX_OBSERVER_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("fused_activation_min_max_observer")                             \
      .SetCreateFn<GpuFusedActivationMinMaxObserverKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t { return 128 * 1024 * 1024; })

REGISTER_FUSED_ACTIVATION_MIN_MAX_OBSERVER_KERNEL(float);
REGISTER_FUSED_ACTIVATION_MIN_MAX_OBSERVER_KERNEL(double);
REGISTER_FUSED_ACTIVATION_MIN_MAX_OBSERVER_KERNEL(half);

}  // namespace oneflow
