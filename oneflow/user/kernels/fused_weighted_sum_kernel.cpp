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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"

namespace oneflow {

namespace {

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
    CHECK_GE(arity, 1);
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
    T* out_ptr = out->mut_dptr<T>();
    auto* cpu_stream = ctx->stream()->As<ep::CpuStream>();
    cpu_stream->ParallelFor(0, shape.elem_cnt(), [&](int64_t s, int64_t e) {
      for (int64_t i = s; i < e; ++i) {
        T out = static_cast<T>(0.0);
        for (int j = 0; j < arity; ++j) { out += inputs[j][i] * static_cast<T>(weights[j]); }
        out_ptr[i] = out * static_cast<T>(alpha);
      }
    });
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_FUSED_WEIGHT_SUM_KERNEL(data_type, cpp_type)         \
  REGISTER_USER_KERNEL("fused_weighted_sum")                          \
      .SetCreateFn<FusedWeightedSumKernel<cpp_type>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("out", 0) == data_type))

REGISTER_FUSED_WEIGHT_SUM_KERNEL(DataType::kDouble, double);
REGISTER_FUSED_WEIGHT_SUM_KERNEL(DataType::kFloat, float);
REGISTER_FUSED_WEIGHT_SUM_KERNEL(DataType::kFloat16, float16);

}  // namespace oneflow
