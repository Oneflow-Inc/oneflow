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
#include "oneflow/core/framework/framework.h"

#include <algorithm>

namespace oneflow {

template<typename T>
void FakeQuantizationPerLayerSymmetric(const T *in_ptr, const T scale,
                                       const int32_t quantization_bit, const int64_t num_elements,
                                       T *out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound;
  FOR_RANGE(int64_t, i, 0, num_elements) {
    T out = std::round(in_ptr[i] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[i] = out * scale;
  }
}

template<typename T>
void FakeQuantizationPerLayerAffine(const T *in_ptr, const T scale, const T zero_point,
                                    const int32_t quantization_bit, const int64_t num_elements,
                                    T *out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
  T lower_bound = 0;
  FOR_RANGE(int64_t, i, 0, num_elements) {
    T out = std::round(in_ptr[i] / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[i] = (out - zero_point) * scale;
  }
}

template<typename T>
void FakeQuantizationPerLayerCambricon(const T *in_ptr, const T shift,
                                       const int32_t quantization_bit, const int64_t num_elements,
                                       T *out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound;
  T scale = static_cast<T>(pow(2.0, static_cast<int32_t>(shift)));
  FOR_RANGE(int64_t, i, 0, num_elements) {
    T out = std::round(in_ptr[i] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[i] = out * scale;
  }
}

template<typename T>
class CpuFakeQuantizationKernel final : public user_op::OpKernel {
 public:
  CpuFakeQuantizationKernel() = default;
  ~CpuFakeQuantizationKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor *scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor *zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);
    user_op::Tensor *out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    const T *in_ptr = in->dptr<T>();
    const T *scale_ptr = scale->dptr<T>();
    T *out_ptr = out->mut_dptr<T>();

    if (quantization_formula == "google") {
      int64_t outer_num = 1;
      int64_t inner_num = in->shape().elem_cnt();
      if (scale->shape().elem_cnt() > 1) {  // per-channel quantization
        outer_num = in->shape().At(0);
        inner_num = in->shape().Count(1);
      }

      if (quantization_scheme == "symmetric") {
        FOR_RANGE(int64_t, c, 0, outer_num) {
          FakeQuantizationPerLayerSymmetric(in_ptr, scale_ptr[c], quantization_bit, inner_num,
                                            out_ptr);
          in_ptr += inner_num;
          out_ptr += inner_num;
        }
      } else {  // quantization_scheme == "affine"
        const T *zero_point_ptr = zero_point->dptr<T>();
        FOR_RANGE(int64_t, c, 0, outer_num) {
          FakeQuantizationPerLayerAffine(in_ptr, scale_ptr[c], zero_point_ptr[c], quantization_bit,
                                         inner_num, out_ptr);
          in_ptr += inner_num;
          out_ptr += inner_num;
        }
      }
    } else if (quantization_formula == "cambricon") {
      FakeQuantizationPerLayerCambricon(in_ptr, scale_ptr[0], quantization_bit,
                                        in->shape().elem_cnt(), out_ptr);
    } else {
      UNIMPLEMENTED();
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FAKE_QUANTIZATION_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("fake_quantization")                          \
      .SetCreateFn<CpuFakeQuantizationKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU) \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_FAKE_QUANTIZATION_KERNEL(float);
REGISTER_FAKE_QUANTIZATION_KERNEL(double);

}  // namespace oneflow
