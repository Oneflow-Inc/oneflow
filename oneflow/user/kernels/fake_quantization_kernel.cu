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
__global__ void FakeQuantizationSymmetric(const T *in_ptr, const T *scale_ptr,
                                          const int64_t scale_size, const int64_t elements,
                                          const int64_t panel_size, const double quantization_bit,
                                          T *out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound;

  while (gid < elements) {
    int64_t channel_index = gid / panel_size;
    int64_t scale_idx = min(scale_size - 1, channel_index);

    T scale = scale_ptr[scale_idx];

    T out = round(in_ptr[gid] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out * scale;

    gid += step;
  }
}

template<typename T>
__global__ void FakeQuantizationAffine(const T *in_ptr, const T *scale_ptr, const T *zero_point_ptr,
                                       const int64_t scale_size, const int64_t elements,
                                       const int64_t panel_size, const double quantization_bit,
                                       T *out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
  T lower_bound = 0;

  while (gid < elements) {
    int64_t channel_index = gid / panel_size;
    int64_t scale_idx = min(scale_size - 1, channel_index);

    T scale = scale_ptr[scale_idx];
    T zero_point = zero_point_ptr[scale_idx];

    T out = round(in_ptr[gid] / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = (out - zero_point) * scale;

    gid += step;
  }
}

template<typename T>
__global__ void FakeQuantizationCambricon(const T *in_ptr, const T *shift, const int64_t scale_size,
                                          const int64_t elements, const int64_t panel_size,
                                          const double quantization_bit, T *out_ptr) {
  int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;

  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound;

  T scale = static_cast<T>(pow(2.0, static_cast<int32_t>(shift[0])));

  while (gid < elements) {
    T out = round(in_ptr[gid] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out * scale;
    gid += step;
  }
}

}  // namespace

template<typename T>
class GpuFakeQuantizationKernel final : public user_op::OpKernel {
 public:
  GpuFakeQuantizationKernel() = default;
  ~GpuFakeQuantizationKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor *scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor *zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);
    user_op::Tensor *out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    const int64_t elements = in->shape().elem_cnt();
    const int64_t panel_size = in->shape().Count(1);
    const int64_t scale_size = scale->shape().elem_cnt();

    if (quantization_formula == "google") {
      if (quantization_scheme == "symmetric") {
        RUN_CUDA_KERNEL((FakeQuantizationSymmetric<T>), ctx->device_ctx(), elements, in->dptr<T>(),
                        scale->dptr<T>(), scale_size, elements, panel_size, quantization_bit,
                        out->mut_dptr<T>());
      } else {  // quantization_scheme == "affine"
        RUN_CUDA_KERNEL((FakeQuantizationAffine<T>), ctx->device_ctx(), elements, in->dptr<T>(),
                        scale->dptr<T>(), zero_point->dptr<T>(), scale_size, elements, panel_size,
                        quantization_bit, out->mut_dptr<T>());
      }
    } else if (quantization_formula == "cambricon") {
      RUN_CUDA_KERNEL((FakeQuantizationCambricon<T>), ctx->device_ctx(), elements, in->dptr<T>(),
                      scale->dptr<T>(), scale_size, elements, panel_size, quantization_bit,
                      out->mut_dptr<T>());
    } else {
      UNIMPLEMENTED();
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FAKE_QUANTIZATION_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("fake_quantization")                          \
      .SetCreateFn<GpuFakeQuantizationKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU) \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_FAKE_QUANTIZATION_KERNEL(float);
REGISTER_FAKE_QUANTIZATION_KERNEL(double);

}  // namespace oneflow
