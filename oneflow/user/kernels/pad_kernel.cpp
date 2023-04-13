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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/constant_pad.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::ConstantPad> NewConstantPadPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("y", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::ConstantPadFactory>(ctx->device_type(),
                                                                        data_type);
}

auto ConstantPadPrimitiveExists() {
  return hob::make_custom("ConstantPadPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewConstantPadPrimitive(&ctx).operator bool();
  });
}

}  // namespace

class PadKernel final : public OpKernel, public CudaGraphSupport {
 public:
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    if (y->shape_view().NumAxes() > 0 && y->shape_view().elem_cnt() == 0) {
      // if output is 0-shape tensor, than do nothing and return
      return;
    }

    Scalar value;
    if (IsIntegralDataType(x->data_type()) || x->data_type() == kBool) {
      value = Scalar(ctx->Attr<int64_t>("integral_constant_value"));
    } else {
      value = Scalar(ctx->Attr<double>("floating_constant_value"));
    }

    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
    const int64_t ndims = x->shape_view().NumAxes();
    CHECK_EQ(padding_before.size(), ndims);

    std::unique_ptr<ep::primitive::ConstantPad> pad_primitive = NewConstantPadPrimitive(ctx);
    CHECK(pad_primitive);

    pad_primitive->Launch(ctx->stream(), ndims, x->shape_view().ptr(), x->dptr(),
                          padding_before.data(), padding_after.data(), value, y->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("pad").SetCreateFn<PadKernel>().SetIsMatchedHob(ConstantPadPrimitiveExists());

}  // namespace user_op

}  // namespace oneflow
