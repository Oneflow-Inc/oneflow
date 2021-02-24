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
void GenQuantScalePerLayerSymmetric(const T *in, const int64_t current_train_step,
                                    const int64_t stop_update_after_iters, const bool is_training,
                                    const int32_t quantization_bit, const int64_t num_elements,
                                    const float momentum, T *moving_max, T *moving_min, T *scale,
                                    T *zero_point) {
  if (current_train_step <= stop_update_after_iters && is_training) {
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
  }

  T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  *scale = (*moving_max) / denominator;
  *zero_point = 0;
}

template<typename T>
void GenQuantScalePerLayerAffine(const T *in, const int64_t current_train_step,
                                 const int64_t stop_update_after_iters, const bool is_training,
                                 const int32_t quantization_bit, const int64_t num_elements,
                                 const float momentum, T *moving_max, T *moving_min, T *scale,
                                 T *zero_point) {
  if (current_train_step <= stop_update_after_iters && is_training) {
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
  }

  T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;
  *scale = ((*moving_max) - (*moving_min)) / denominator;
  *zero_point = -(*moving_min) / (*scale);
}

template<typename T>
void GenQuantScalePerLayerCambricon(const T *in, const int64_t current_train_step,
                                    const int64_t stop_update_after_iters, const bool is_training,
                                    const int32_t quantization_bit, const int64_t num_elements,
                                    const float momentum, T *moving_max, T *moving_min, T *scale,
                                    T *zero_point) {
  if (current_train_step <= stop_update_after_iters && is_training) {
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
  }

  *scale = std::floor(std::log2(*moving_max)) - (quantization_bit - 2);
  *zero_point = 0;
}

template<typename T>
class CpuMovingAverageMinMaxObserverKernel final : public user_op::OpKernel {
 public:
  CpuMovingAverageMinMaxObserverKernel() = default;
  ~CpuMovingAverageMinMaxObserverKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor *current_train_step =
        ctx->Tensor4ArgNameAndIndex("current_train_step", 0);
    user_op::Tensor *moving_max = ctx->Tensor4ArgNameAndIndex("moving_max", 0);
    user_op::Tensor *moving_min = ctx->Tensor4ArgNameAndIndex("moving_min", 0);
    user_op::Tensor *scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor *zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);

    const std::string quantization_scheme = ctx->Attr<std::string>("quantization_scheme");
    const int32_t quantization_bit = ctx->Attr<int32_t>("quantization_bit");
    const float momentum = ctx->Attr<float>("momentum");
    const int64_t stop_update_after_iters = ctx->Attr<int64_t>("stop_update_after_iters");
    const bool is_training = ctx->Attr<bool>("training");
    const std::string quantization_formula = ctx->Attr<std::string>("quantization_formula");

    const T *in_ptr = in->dptr<T>();
    const int64_t *current_train_step_ptr = current_train_step->dptr<int64_t>();
    T *moving_max_ptr = moving_max->mut_dptr<T>();
    T *moving_min_ptr = moving_min->mut_dptr<T>();
    T *scale_ptr = scale->mut_dptr<T>();
    T *zero_point_ptr = zero_point->mut_dptr<T>();

    int64_t num_elements = in->shape().elem_cnt();

    if (quantization_formula == "google") {
      if (quantization_scheme == "symmetric") {
        GenQuantScalePerLayerSymmetric(in_ptr, *current_train_step_ptr, stop_update_after_iters,
                                       is_training, quantization_bit, num_elements, momentum,
                                       moving_max_ptr, moving_min_ptr, scale_ptr, zero_point_ptr);
      } else {  // quantization_scheme == "affine"
        GenQuantScalePerLayerAffine(in_ptr, *current_train_step_ptr, stop_update_after_iters,
                                    is_training, quantization_bit, num_elements, momentum,
                                    moving_max_ptr, moving_min_ptr, scale_ptr, zero_point_ptr);
      }
    } else if (quantization_formula == "cambricon") {
      GenQuantScalePerLayerCambricon(in_ptr, *current_train_step_ptr, stop_update_after_iters,
                                     is_training, quantization_bit, num_elements, momentum,
                                     moving_max_ptr, moving_min_ptr, scale_ptr, zero_point_ptr);
    } else {
      UNIMPLEMENTED();
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
