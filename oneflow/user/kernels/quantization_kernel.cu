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

namespace oneflow {

namespace {

template<typename T>
__global__ void QuantizationSymmetric(const T* in_ptr, const T* scale_ptr, const int64_t scale_size,
                                      const int64_t elements, const int64_t panel_size,
                                      const int32_t quantization_bit, T* out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  float upper_bound = pow(2.0, quantization_bit - 1) - 1;
  float lower_bound = -upper_bound - 1;

  while (gid < elements) {
    int64_t channel_index = gid / panel_size;
    int64_t scale_idx = min(scale_size - 1, channel_index);

    float scale = scale_ptr[scale_idx];
    float in = in_ptr[gid];

    float out = nearbyint(in / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = static_cast<T>(out);

    gid += step;
  }
}

template<typename T>
__global__ void QuantizationAffine(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                   const int64_t scale_size, const int64_t elements,
                                   const int64_t panel_size, const int32_t quantization_bit,
                                   T* out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  float upper_bound = pow(2.0, quantization_bit) - 1;
  float lower_bound = 0;

  while (gid < elements) {
    int64_t channel_index = gid / panel_size;
    int64_t scale_idx = min(scale_size - 1, channel_index);

    float scale = scale_ptr[scale_idx];
    float zero_point = zero_point_ptr[scale_idx];
    float in = in_ptr[gid];

    float out = nearbyint(in / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = static_cast<T>(out);

    gid += step;
  }
}

template<typename T>
__global__ void QuantizationCambricon(const T* in_ptr, const T* shift, const int64_t scale_size,
                                      const int64_t elements, const int64_t panel_size,
                                      const double quantization_bit, T* out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  float upper_bound = pow(2.0, quantization_bit - 1) - 1;
  float lower_bound = -upper_bound - 1;

  float scale = pow(2.0, static_cast<int32_t>(shift[0]));

  while (gid < elements) {
    float in = in_ptr[gid];
    float out = nearbyint(in / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = static_cast<T>(out);
    gid += step;
  }
}

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

template<int pack_size, typename T, typename OutT>
__global__ void OFPerTensorQuantizationSymmetric(const int64_t elements, const T* in_ptr,
                                                 const T* scale_ptr, const OutT upper_bound,
                                                 const OutT lower_bound, OutT* out_ptr) {
  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  using StoreType = cuda::elementwise::PackType<OutT, pack_size>;
  using StorePack = cuda::elementwise::Pack<OutT, pack_size>;

  int64_t tid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x * pack_size;

  float scale = *scale_ptr;

  for (int64_t idx = tid * pack_size; idx < elements; idx += step) {
    StorePack out;
    LoadPack in;
    in.storage = reinterpret_cast<const LoadType*>(in_ptr + idx)[0];
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      out.elem[i] = max(min(__float2int_rn(static_cast<float>(in.elem[i]) / scale), upper_bound),
                        lower_bound);
    }
    reinterpret_cast<StoreType*>(out_ptr + idx)[0] = out.storage;
  }

  int rest = ModDiv<pack_size>(elements);

  if (rest > 0 && tid == (gridDim.x * blockDim.x - 1)) {
    in_ptr += elements - rest;
    out_ptr += elements - rest;
#pragma unroll
    for (int i = 0; i < rest; ++i) {
      out_ptr[i] =
          max(min(__float2int_rn(static_cast<float>(in_ptr[i]) / scale), upper_bound), lower_bound);
    }
  }
}

template<int pack_size, typename T, typename OutT>
__global__ void OFPerTensorQuantizationAffine(const int64_t elements, const T* in_ptr,
                                              const T* scale_ptr, const OutT* zero_point_ptr,
                                              const OutT upper_bound, const OutT lower_bound,
                                              OutT* out_ptr) {
  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  using StoreType = cuda::elementwise::PackType<OutT, pack_size>;
  using StorePack = cuda::elementwise::Pack<OutT, pack_size>;

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
#pragma unroll
    for (int i = 0; i < rest; ++i) {
      out_ptr[i] =
          max(min(__float2int_rn(static_cast<float>(in_ptr[i]) / scale + zero_point), upper_bound),
              lower_bound);
    }
  }
}

template<typename T, typename OutT>
void ApplyOFPerTensorQuantization(user_op::KernelComputeContext* ctx,
                                  const std::string& quantization_scheme,
                                  const int32_t quantization_bit, const user_op::Tensor* in,
                                  const user_op::Tensor* scale, const user_op::Tensor* zero_point,
                                  user_op::Tensor* out) {
  constexpr int pack_size = cuda::elementwise::PackSize<T>();

  const int64_t elements = in->shape_view().elem_cnt();
  int64_t pack_num = (elements + pack_size - 1) / pack_size;
  int grid_size;
  cuda::elementwise::GetNumBlocks(pack_num, &grid_size);

  OutT upper_bound = static_cast<OutT>(pow(2.0, quantization_bit - 1)) - 1;
  OutT lower_bound = -upper_bound - 1;
  auto stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
  if (quantization_scheme == "symmetric") {
    OFPerTensorQuantizationSymmetric<pack_size, T, OutT><<<grid_size, cuda::elementwise::kBlockSize, 0, stream>>>(
        elements, in->dptr<T>(), scale->dptr<T>(), upper_bound, lower_bound, out->mut_dptr<OutT>());
  } else {
    OFPerTensorQuantizationAffine<pack_size, T, OutT><<<grid_size, cuda::elementwise::kBlockSize, 0, stream>>>(
        elements, in->dptr<T>(), scale->dptr<T>(), zero_point->dptr<OutT>(), upper_bound,
        lower_bound, out->mut_dptr<OutT>());
  }
}

}  // namespace

template<typename T>
class GpuQuantizationKernel final : public user_op::OpKernel {
 public:
  GpuQuantizationKernel() = default;
  ~GpuQuantizationKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor* zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    const int64_t elements = in->shape_view().elem_cnt();
    const int64_t panel_size = in->shape_view().Count(1);
    const int64_t scale_size = scale->shape_view().elem_cnt();

    // round to even
    auto origin_round_mode = std::fegetround();
    std::fesetround(FE_TONEAREST);

    if (quantization_formula == "oneflow") {
      CHECK_EQ(scale_size, 1)
          << "only support per-tensor quantization for oneflow quantization formula";
      if (quantization_bit == 8) {
        ApplyOFPerTensorQuantization<T, int8_t>(ctx, quantization_scheme, quantization_bit, in,
                                                scale, zero_point, out);
      } else {
        UNIMPLEMENTED();
      }
    } else if (quantization_formula == "google") {
      if (quantization_scheme == "symmetric") {
        RUN_CUDA_KERNEL((QuantizationSymmetric<T>), ctx->stream(), elements, in->dptr<T>(),
                        scale->dptr<T>(), scale_size, elements, panel_size, quantization_bit,
                        out->mut_dptr<T>());
      } else {  // quantization_scheme == "affine"
        RUN_CUDA_KERNEL((QuantizationAffine<T>), ctx->stream(), elements, in->dptr<T>(),
                        scale->dptr<T>(), zero_point->dptr<T>(), scale_size, elements, panel_size,
                        quantization_bit, out->mut_dptr<T>());
      }
    } else if (quantization_formula == "cambricon") {
      RUN_CUDA_KERNEL((QuantizationCambricon<T>), ctx->stream(), elements, in->dptr<T>(),
                      scale->dptr<T>(), scale_size, elements, panel_size, quantization_bit,
                      out->mut_dptr<T>());
    } else {
      UNIMPLEMENTED();
    }

    std::fesetround(origin_round_mode);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_QUANTIZATION_KERNEL(dtype)                            \
  REGISTER_USER_KERNEL("quantization")                                 \
      .SetCreateFn<GpuQuantizationKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_QUANTIZATION_KERNEL(float);
REGISTER_QUANTIZATION_KERNEL(double);
REGISTER_QUANTIZATION_KERNEL(half);

}  // namespace oneflow
