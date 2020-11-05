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
void GenQuantScalePerLayerSymmetric(const int32_t quantize_to_bit, const int64_t num_elements,
                                    const float momentum, const T *activation, T *moving_max,
                                    T *moving_min, T *scale, T *zero_point) {
  T activation_max = *std::max_element(activation, activation + num_elements);
  T activation_min = *std::min_element(activation, activation + num_elements);

  activation_max = std::max(std::abs(activation_max), std::abs(activation_min));

  T moving_max_val = moving_max[0];

  if (moving_max_val == 0)
    moving_max[0] = activation_max;
  else
    moving_max[0] = moving_max_val * momentum + activation_max * (1 - momentum);

  // NOTE(Liang Depeng): symmetric quantization only use moving_max to calculate the scale
  moving_min[0] = moving_max[0];

  T denominator = T(pow(2.0, quantize_to_bit - 1)) - 1;
  scale[0] = moving_max[0] / denominator;
  zero_point[0] = 0;
}

template<typename T>
void GenQuantScalePerLayerAffine(const int32_t quantize_to_bit, const int64_t num_elements,
                                 const float momentum, const T *activation, T *moving_max,
                                 T *moving_min, T *scale, T *zero_point) {
  T activation_max = *std::max_element(activation, activation + num_elements);
  T activation_min = *std::min_element(activation, activation + num_elements);

  T moving_max_val = moving_max[0];
  if (moving_max_val == 0)
    moving_max[0] = activation_max;
  else
    moving_max[0] = moving_max_val * momentum + activation_max * (1 - momentum);

  T moving_min_val = moving_min[0];
  if (moving_min_val == 0)
    moving_min[0] = activation_min;
  else
    moving_min[0] = moving_min_val * momentum + activation_min * (1 - momentum);

  T denominator = T(pow(2.0, quantize_to_bit)) - 1;
  scale[0] = (moving_max[0] - moving_min[0]) / denominator;
  zero_point[0] = -moving_min[0] / scale[0];
}

template<typename T>
class CpuGenerateQuantizeScaleForActivationKernel final : public user_op::OpKernel {
 public:
  CpuGenerateQuantizeScaleForActivationKernel() = default;
  ~CpuGenerateQuantizeScaleForActivationKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *activation = ctx->Tensor4ArgNameAndIndex("activation", 0);
    user_op::Tensor *moving_max = ctx->Tensor4ArgNameAndIndex("moving_max", 0);
    user_op::Tensor *moving_min = ctx->Tensor4ArgNameAndIndex("moving_min", 0);
    user_op::Tensor *scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor *zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);

    const std::string quantizer_type = ctx->Attr<std::string>("quantizer_type");
    const int32_t quantize_to_bit = ctx->Attr<int32_t>("quantize_to_bit");
    const float momentum = ctx->Attr<float>("momentum");

    const T *activation_ptr = activation->dptr<T>();
    T *moving_max_ptr = moving_max->mut_dptr<T>();
    T *moving_min_ptr = moving_min->mut_dptr<T>();
    T *scale_ptr = scale->mut_dptr<T>();
    T *zero_point_ptr = zero_point->mut_dptr<T>();

    int64_t num_elements = activation->shape().elem_cnt();

    if (quantizer_type == "symmetric") {
      GenQuantScalePerLayerSymmetric(quantize_to_bit, num_elements, momentum, activation_ptr,
                                     moving_max_ptr, moving_min_ptr, scale_ptr, zero_point_ptr);
    } else {  // quantizer_type == "affine"
      GenQuantScalePerLayerAffine(quantize_to_bit, num_elements, momentum, activation_ptr,
                                  moving_max_ptr, moving_min_ptr, scale_ptr, zero_point_ptr);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GENERATE_QUANTIZE_SCALE_FOR_ACTIVATION_KERNEL(dtype)    \
  REGISTER_USER_KERNEL("generate_quantize_scale_for_activation")         \
      .SetCreateFn<CpuGenerateQuantizeScaleForActivationKernel<dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)     \
                       & (user_op::HobDataType("activation", 0) == GetDataType<dtype>::value))

REGISTER_GENERATE_QUANTIZE_SCALE_FOR_ACTIVATION_KERNEL(float);
REGISTER_GENERATE_QUANTIZE_SCALE_FOR_ACTIVATION_KERNEL(double);

}  // namespace oneflow
