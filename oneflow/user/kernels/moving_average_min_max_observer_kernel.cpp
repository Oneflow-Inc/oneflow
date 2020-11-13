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
void GenQuantScalePerLayerSymmetric(const T *in, const int32_t quantize_to_bit,
                                    const int64_t num_elements, const float momentum, T *moving_max,
                                    T *moving_min, T *scale, T *zero_point) {
  T in_max = *std::max_element(in, in + num_elements);
  T in_min = *std::min_element(in, in + num_elements);

  in_max = std::max(std::abs(in_max), std::abs(in_min));

  T moving_max_val = *moving_max;

  if (moving_max_val == 0) {
    *moving_max = in_max;
  } else {
    *moving_max = moving_max_val * momentum + in_max * (1 - momentum);
  }

  // NOTE(Liang Depeng): symmetric quantization only use moving_max to calculate the scale
  *moving_min = *moving_max;

  T denominator = static_cast<T>(pow(2.0, quantize_to_bit - 1)) - 1;
  *scale = (*moving_max) / denominator;
  *zero_point = 0;
}

template<typename T>
void GenQuantScalePerLayerAffine(const T *in, const int32_t quantize_to_bit,
                                 const int64_t num_elements, const float momentum, T *moving_max,
                                 T *moving_min, T *scale, T *zero_point) {
  T in_max = *std::max_element(in, in + num_elements);
  T in_min = *std::min_element(in, in + num_elements);

  T moving_max_val = *moving_max;
  if (moving_max_val == 0) {
    *moving_max = in_max;
  } else {
    *moving_max = moving_max_val * momentum + in_max * (1 - momentum);
  }

  T moving_min_val = *moving_min;
  if (moving_min_val == 0) {
    *moving_min = in_min;
  } else {
    *moving_min = moving_min_val * momentum + in_min * (1 - momentum);
  }

  T denominator = static_cast<T>(pow(2.0, quantize_to_bit)) - 1;
  *scale = ((*moving_max) - (*moving_min)) / denominator;
  *zero_point = -(*moving_min) / (*scale);
}

template<typename T>
class CpuMovingAverageMinMaxObserverKernel final : public user_op::OpKernel {
 public:
  CpuMovingAverageMinMaxObserverKernel() = default;
  ~CpuMovingAverageMinMaxObserverKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *moving_max = ctx->Tensor4ArgNameAndIndex("moving_max", 0);
    user_op::Tensor *moving_min = ctx->Tensor4ArgNameAndIndex("moving_min", 0);
    user_op::Tensor *scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor *zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);

    const std::string quantize_scheme = ctx->Attr<std::string>("quantize_scheme");
    const int32_t quantize_to_bit = ctx->Attr<int32_t>("quantize_to_bit");
    const float momentum = ctx->Attr<float>("momentum");

    const T *in_ptr = in->dptr<T>();
    T *moving_max_ptr = moving_max->mut_dptr<T>();
    T *moving_min_ptr = moving_min->mut_dptr<T>();
    T *scale_ptr = scale->mut_dptr<T>();
    T *zero_point_ptr = zero_point->mut_dptr<T>();

    int64_t num_elements = in->shape().elem_cnt();

    if (quantize_scheme == "symmetric") {
      GenQuantScalePerLayerSymmetric(in_ptr, quantize_to_bit, num_elements, momentum,
                                     moving_max_ptr, moving_min_ptr, scale_ptr, zero_point_ptr);
    } else {  // quantize_scheme == "affine"
      GenQuantScalePerLayerAffine(in_ptr, quantize_to_bit, num_elements, momentum, moving_max_ptr,
                                  moving_min_ptr, scale_ptr, zero_point_ptr);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MOVING_AVERAGE_MIN_MAX_OBSERVER_KERNEL(dtype)       \
  REGISTER_USER_KERNEL("moving_average_min_max_observer")            \
      .SetCreateFn<CpuMovingAverageMinMaxObserverKernel<dtype>>()    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU) \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_MOVING_AVERAGE_MIN_MAX_OBSERVER_KERNEL(float);
REGISTER_MOVING_AVERAGE_MIN_MAX_OBSERVER_KERNEL(double);

}  // namespace oneflow
