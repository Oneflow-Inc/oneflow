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
#include <complex>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/fill.h"

namespace oneflow {
namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Fill> NewFillPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->device_type(), data_type);
}

class ConstantKernel final : public OpKernel {
 public:
  ConstantKernel() = default;
  ~ConstantKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool is_complex_value = ctx->Attr<bool>("is_complex_value");
    bool is_floating_value = ctx->Attr<bool>("is_floating_value");

    const Scalar value = is_complex_value
                             ? Scalar(ctx->Attr<std::complex<double>>("complex_value"))
                             : (is_floating_value ? Scalar(ctx->Attr<double>("floating_value"))
                                                  : Scalar(ctx->Attr<int64_t>("integer_value")));
    const int64_t elem_cnt = out_tensor->shape_view().elem_cnt();
    CHECK_GE(elem_cnt, 0);
    if (elem_cnt == 0) { return; }
    std::unique_ptr<ep::primitive::Fill> fill = NewFillPrimitive(ctx);
    CHECK(fill);
    fill->Launch(ctx->stream(), out_tensor->mut_dptr(), value, elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto FillPrimitiveExists() {
  return hob::make_custom("FillPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewFillPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("constant")
    .SetCreateFn<ConstantKernel>()
    .SetIsMatchedHob(FillPrimitiveExists() == true);

}  // namespace

}  // namespace user_op
}  // namespace oneflow
