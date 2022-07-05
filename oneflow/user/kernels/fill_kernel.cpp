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
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_unary.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Fill> NewFillPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->device_type(), data_type);
}

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseUnary> NewBroadcastUnaryPrimitive(Context* ctx) {
  const user_op::TensorDesc* src = ctx->TensorDesc4ArgNameAndIndex("value", 0);
  const user_op::TensorDesc* dst = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseUnaryFactory>(
      ctx->device_type(), ep::primitive::UnaryOp::kIdentity, src->data_type(), dst->data_type(), dst->shape().NumAxes());
}

}  // namespace

auto FillPrimitiveExists() {
  return hob::make_custom("FillPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewFillPrimitive(&ctx).operator bool();
  });
}

auto BroadcastUnaryPrimitiveExists() {
  return hob::make_custom("BroadcastElementwiseUnaryPrimitiveExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            return NewBroadcastUnaryPrimitive(&ctx).operator bool();
                          });
}

class FillKernel final : public user_op::OpKernel {
 public:
  FillKernel() = default;
  ~FillKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const bool is_floating_value = ctx->Attr<bool>("is_floating_value");
    const Scalar value = is_floating_value ? Scalar(ctx->Attr<double>("floating_value"))
                                           : Scalar(ctx->Attr<int64_t>("integral_value"));
    const int32_t elem_cnt = in->shape_view().elem_cnt();
    CHECK_GE(elem_cnt, 0);
    if (elem_cnt == 0) { return; }
    auto fill_primitive = NewFillPrimitive(ctx);
    CHECK(fill_primitive);
    fill_primitive->Launch(ctx->stream(), out->mut_dptr(), value, elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class FillTensorKernel final : public user_op::OpKernel {
 public:
  FillTensorKernel() = default;
  ~FillTensorKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);

    auto primitive = NewBroadcastUnaryPrimitive(ctx);
    CHECK(primitive);

    if(value->shape_view().elem_cnt() != 0) {
      primitive->Launch(ctx->stream(), value->shape_view().NumAxes(), value->shape_view().data(), value->dptr(),
                        out->shape_view().NumAxes(), out->shape_view().data(), out->mut_dptr());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("fill_tensor_")
    .SetCreateFn<FillTensorKernel>()
    .SetIsMatchedHob(BroadcastUnaryPrimitiveExists());

REGISTER_USER_KERNEL("fill_").SetCreateFn<FillKernel>().SetIsMatchedHob(FillPrimitiveExists()
                                                                        == true);

}  // namespace oneflow
