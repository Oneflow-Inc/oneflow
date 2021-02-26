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
void GenQuantScaleSymmetric(const T *in_ptr, const int32_t quantization_bit,
                            const int64_t num_elements, T *scale, T *zero_point) {
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

  in_max = std::max(std::abs(in_max), std::abs(in_min));

  T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;

  *scale = in_max / denominator;
  *zero_point = 0;
}

template<typename T>
void GenQuantScaleAffine(const T *in_ptr, const int32_t quantization_bit,
                         const int64_t num_elements, T *scale, T *zero_point) {
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

  T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;

  *scale = (in_max - in_min) / denominator;
  *zero_point = -in_min / (*scale);
}

template<typename T>
void GenQuantScaleCambricon(const T *in_ptr, const int32_t quantization_bit,
                            const int64_t num_elements, T *scale, T *zero_point) {
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

  in_max = std::max(std::abs(in_max), std::abs(in_min));

  *scale = std::floor(std::log2(in_max)) - (quantization_bit - 2);
  *zero_point = 0;
}

template<typename T>
class CpuMinMaxObserverKernel final : public user_op::OpKernel {
 public:
  CpuMinMaxObserverKernel() = default;
  ~CpuMinMaxObserverKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor *zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const bool per_layer_quantization = ctx->Attr<bool>("per_layer_quantization");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    const T *in_ptr = in->dptr<T>();
    T *scale_ptr = scale->mut_dptr<T>();
    T *zero_point_ptr = zero_point->mut_dptr<T>();

    if (quantization_formula == "google") {
      // NOTE(Liang Depeng): per-layer quantization by default
      int64_t outer_num = 1;
      int64_t inner_num = in->shape().elem_cnt();
      if (!per_layer_quantization) {  // per-channel quantization
        outer_num = in->shape().At(0);
        inner_num = in->shape().Count(1);
      }

      if (quantization_scheme == "symmetric") {
        FOR_RANGE(int64_t, c, 0, outer_num) {
          GenQuantScaleSymmetric(in_ptr, quantization_bit, inner_num, scale_ptr, zero_point_ptr);
          in_ptr += inner_num;
          scale_ptr += 1;
          zero_point_ptr += 1;
        }
      } else {  // quantization_scheme == "affine"
        FOR_RANGE(int64_t, c, 0, outer_num) {
          GenQuantScaleAffine(in_ptr, quantization_bit, inner_num, scale_ptr, zero_point_ptr);
          in_ptr += inner_num;
          scale_ptr += 1;
          zero_point_ptr += 1;
        }
      }
    } else if (quantization_formula == "cambricon") {
      GenQuantScaleCambricon(in_ptr, quantization_bit, in->shape().elem_cnt(), scale_ptr,
                             zero_point_ptr);
    } else {
      UNIMPLEMENTED();
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MIN_MAX_OBSERVER_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("min_max_observer")                           \
      .SetCreateFn<CpuMinMaxObserverKernel<dtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU) \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_MIN_MAX_OBSERVER_KERNEL(float);
REGISTER_MIN_MAX_OBSERVER_KERNEL(double);

}  // namespace oneflow
