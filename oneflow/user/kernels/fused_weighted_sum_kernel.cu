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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, int arity>
struct Params {
  const T* inputs[arity];
  float weights[arity];
  float alpha{};
  T* output;
  int64_t n;
};

template<typename T, int arity, bool acc>
__global__ void WeightedSumKernel(Params<T, arity> params) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, params.n) {
    T out = 0;
    if (acc) { out = params.output[i]; }
#pragma unroll
    for (int j = 0; j < arity; ++j) {
      out += params.inputs[j][i] * static_cast<T>(params.weights[j]);
    }
    params.output[i] = out * static_cast<T>(params.alpha);
  }
}

template<typename T, int arity, bool acc>
void LaunchWeightedSum(ep::Stream* stream, int n, const T** inputs, const float* weights,
                       float alpha, T* output) {
  Params<T, arity> params{};
  for (int i = 0; i < arity; ++i) {
    params.inputs[i] = *(inputs + i);
    params.weights[i] = *(weights + i);
  }
  params.alpha = alpha;
  params.output = output;
  params.n = n;
  RUN_CUDA_KERNEL((WeightedSumKernel<T, arity, acc>), stream, n, params);
}

template<typename T, bool acc>
void DispatchWeightedSum(ep::Stream* stream, int arity, int64_t n, const T** inputs,
                         const float* weights, float alpha, T* output) {
  if (arity == 1) {
    LaunchWeightedSum<T, 1, acc>(stream, n, inputs, weights, alpha, output);
  } else if (arity == 2) {
    LaunchWeightedSum<T, 2, acc>(stream, n, inputs, weights, alpha, output);
  } else if (arity == 3) {
    LaunchWeightedSum<T, 3, acc>(stream, n, inputs, weights, alpha, output);
  } else if (arity == 4) {
    LaunchWeightedSum<T, 4, acc>(stream, n, inputs, weights, alpha, output);
  } else if (arity == 5) {
    LaunchWeightedSum<T, 5, acc>(stream, n, inputs, weights, alpha, output);
  } else if (arity == 6) {
    LaunchWeightedSum<T, 6, acc>(stream, n, inputs, weights, alpha, output);
  } else if (arity == 7) {
    LaunchWeightedSum<T, 7, acc>(stream, n, inputs, weights, alpha, output);
  } else if (arity == 8) {
    LaunchWeightedSum<T, 8, acc>(stream, n, inputs, weights, alpha, output);
  } else if (arity > 8) {
    LaunchWeightedSum<T, 8, acc>(stream, n, inputs, weights, 1.0F, output);
    DispatchWeightedSum<T, true>(stream, arity - 8, n, inputs + 8, weights + 8, alpha, output);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class FusedWeightedSumKernel final : public user_op::OpKernel {
 public:
  FusedWeightedSumKernel() = default;
  ~FusedWeightedSumKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t arity = ctx->input_size("in");
    CHECK_GE(arity, 1) << "input_size should be greater than 0.";
    const std::vector<float>& weights = ctx->Attr<std::vector<float>>("weights");
    CHECK_EQ(weights.size(), arity);
    const float alpha = ctx->Attr<float>("alpha");
    const DataType data_type = out->data_type();
    const ShapeView& shape = out->shape_view();
    std::vector<const T*> inputs(arity);
    for (int i = 0; i < arity; ++i) {
      const user_op::Tensor* in_i = ctx->Tensor4ArgNameAndIndex("in", i);
      CHECK(in_i->shape_view() == shape);
      CHECK_EQ(in_i->data_type(), data_type);
      inputs[i] = in_i->dptr<T>();
    }
    DispatchWeightedSum<T, false>(ctx->stream(), arity, shape.elem_cnt(), inputs.data(),
                                  weights.data(), alpha, out->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_FUSED_WEIGHT_SUM_KERNEL(data_type, cpp_type)          \
  REGISTER_USER_KERNEL("fused_weighted_sum")                           \
      .SetCreateFn<FusedWeightedSumKernel<cpp_type>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == data_type))

REGISTER_FUSED_WEIGHT_SUM_KERNEL(DataType::kDouble, double);
REGISTER_FUSED_WEIGHT_SUM_KERNEL(DataType::kFloat, float);
REGISTER_FUSED_WEIGHT_SUM_KERNEL(DataType::kFloat16, half);

}  // namespace oneflow
