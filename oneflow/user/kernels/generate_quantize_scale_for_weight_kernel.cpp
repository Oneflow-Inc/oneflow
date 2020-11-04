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
void gen_quant_scale_per_layer_symmetric(const int32_t quantize_to_bit, const int64_t num_elements,
                                         const T *weight_ptr, T *weight_scale, T *zero_point) {
  T weight_max = *std::max_element(weight_ptr, weight_ptr + num_elements);
  T weight_min = *std::min_element(weight_ptr, weight_ptr + num_elements);

  weight_max = std::max(std::abs(weight_max), std::abs(weight_min));

  T denominator = T(pow(2.0, quantize_to_bit - 1)) - 1;

  weight_scale[0] = weight_max / denominator;
  zero_point[0] = 0;
}

template<typename T>
void gen_quant_scale_per_layer_affine(const int32_t quantize_to_bit, const int64_t num_elements,
                                      const T *weight_ptr, T *weight_scale, T *zero_point) {
  T weight_max = *std::max_element(weight_ptr, weight_ptr + num_elements);
  T weight_min = *std::min_element(weight_ptr, weight_ptr + num_elements);

  T denominator = T(pow(2.0, quantize_to_bit)) - 1;

  weight_scale[0] = (weight_max - weight_min) / denominator;
  zero_point[0] = -weight_min / weight_scale[0];
}

template<typename T>
class CpuGenerateQuantizeScaleForWeightKernel final : public user_op::OpKernel {
 public:
  CpuGenerateQuantizeScaleForWeightKernel() = default;
  ~CpuGenerateQuantizeScaleForWeightKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor *weight_scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    user_op::Tensor *zero_point = ctx->Tensor4ArgNameAndIndex("zero_point", 0);

    const std::string quantizer_type = ctx->Attr<std::string>("quantizer_type");
    const int32_t quantize_to_bit = ctx->Attr<int32_t>("quantize_to_bit");
    const bool per_layer_quantization = ctx->Attr<bool>("per_layer_quantization");

    const T *weight_ptr = weight->dptr<T>();
    T *weight_scale_ptr = weight_scale->mut_dptr<T>();
    T *zero_point_ptr = zero_point->mut_dptr<T>();

    // NOTE(Liang Depeng): default is per layer quantization
    int64_t outer_num = 1;
    int64_t inner_num = weight->shape().elem_cnt();
    if (!per_layer_quantization) {  // per-channel quantization
      outer_num = weight->shape().At(0);
      inner_num = weight->shape().Count(1);
    }

    if (quantizer_type == "symmetric") {
      FOR_RANGE(int64_t, c, 0, outer_num) {
        gen_quant_scale_per_layer_symmetric(quantize_to_bit, inner_num, weight_ptr,
                                            weight_scale_ptr, zero_point_ptr);
        weight_ptr += inner_num;
        weight_scale_ptr += 1;
        zero_point_ptr += 1;
      }
    } else {  // quantizer_type == "affine"
      FOR_RANGE(int64_t, c, 0, outer_num) {
        gen_quant_scale_per_layer_affine(quantize_to_bit, inner_num, weight_ptr, weight_scale_ptr,
                                         zero_point_ptr);
        weight_ptr += inner_num;
        weight_scale_ptr += 1;
        zero_point_ptr += 1;
      }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GENERATE_QUANTIZE_SCALE_FOR_WEIGHT_KERNEL(dtype)    \
  REGISTER_USER_KERNEL("generate_quantize_scale_for_weight")         \
      .SetCreateFn<CpuGenerateQuantizeScaleForWeightKernel<dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU) \
                       & (user_op::HobDataType("weight", 0) == GetDataType<dtype>::value))

REGISTER_GENERATE_QUANTIZE_SCALE_FOR_WEIGHT_KERNEL(float);
REGISTER_GENERATE_QUANTIZE_SCALE_FOR_WEIGHT_KERNEL(double);

}  // namespace oneflow
