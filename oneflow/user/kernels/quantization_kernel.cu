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

template<typename T, typename OutT>
__global__ void OFPerTensorQuantizationSymmetric(const int64_t elements, const T* in_ptr,
                                                 const T* scale_ptr, const OutT upper_bound,
                                                 const OutT lower_bound, OutT* out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  float scale = *scale_ptr;

  while (gid < elements) {
    float in = in_ptr[gid];
    float out = nearbyint(in / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = static_cast<OutT>(out);

    gid += step;
  }
}

template<typename T, typename OutT>
__global__ void OFPerTensorQuantizationAffine(const int64_t elements, const T* in_ptr,
                                              const T* scale_ptr, const OutT* zero_point_ptr,
                                              const OutT upper_bound, const OutT lower_bound,
                                              OutT* out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  float scale = *scale_ptr;
  float zero_point = *zero_point_ptr;

  while (gid < elements) {
    float in = in_ptr[gid];
    float out = nearbyint(in / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = static_cast<OutT>(out);

    gid += step;
  }
}

struct __align__(8) Half4 {
  half x;
  half y;
  half z;
  half w;
};

struct __align__(4) Byte4 {
  int8_t x;
  int8_t y;
  int8_t z;
  int8_t w;
};

template<>
__global__ void OFPerTensorQuantizationAffine<half, int8_t>(
    const int64_t elements, const half* in_ptr, const half* scale_ptr, const int8_t* zero_point_ptr,
    const int8_t upper_bound, const int8_t lower_bound, int8_t* out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  float scale = *scale_ptr;
  float zero_point = *zero_point_ptr;

  int64_t loops = elements >> 2;
  for (; gid < loops; gid += step) {
    Half4 in = reinterpret_cast<const Half4*>(in_ptr)[gid];
    Byte4 out;
    int x = __float2int_rn(static_cast<float>(in.x) / scale + zero_point);
    int y = __float2int_rn(static_cast<float>(in.y) / scale + zero_point);
    int z = __float2int_rn(static_cast<float>(in.z) / scale + zero_point);
    int w = __float2int_rn(static_cast<float>(in.w) / scale + zero_point);
    out.x = max(min(x, upper_bound), lower_bound);
    out.y = max(min(y, upper_bound), lower_bound);
    out.z = max(min(z, upper_bound), lower_bound);
    out.w = max(min(w, upper_bound), lower_bound);
    reinterpret_cast<Byte4*>(out_ptr)[gid] = out;
  }
  int64_t offset = loops << 2;
  if (offset < elements && gid == loops) {
    for (; offset < elements; offset += 1) {
      float in = in_ptr[offset];
      int out = __float2int_rn(in / scale + zero_point);
      out_ptr[offset] = max(min(out, upper_bound), lower_bound);
    }
  }
}

template<typename T, typename OutT>
void ApplyOFPerTensorQuantization(user_op::KernelComputeContext* ctx,
                                  const std::string& quantization_scheme,
                                  const int32_t quantization_bit, const user_op::Tensor* in,
                                  const user_op::Tensor* scale, const user_op::Tensor* zero_point,
                                  user_op::Tensor* out) {
  const int64_t elements = in->shape_view().elem_cnt();
  OutT upper_bound = static_cast<OutT>(pow(2.0, quantization_bit - 1)) - 1;
  OutT lower_bound = -upper_bound - 1;
  if (quantization_scheme == "symmetric") {
    RUN_CUDA_KERNEL((OFPerTensorQuantizationSymmetric<T, OutT>), ctx->stream(), elements, elements,
                    in->dptr<T>(), scale->dptr<T>(), upper_bound, lower_bound,
                    out->mut_dptr<OutT>());
  } else {
    RUN_CUDA_KERNEL((OFPerTensorQuantizationAffine<T, OutT>), ctx->stream(), elements, elements,
                    in->dptr<T>(), scale->dptr<T>(), zero_point->dptr<OutT>(), upper_bound,
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
