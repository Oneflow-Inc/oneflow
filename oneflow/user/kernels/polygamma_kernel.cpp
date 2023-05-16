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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> NewPrimitive(
    Context* ctx, ep::primitive::BinaryOp binary_op) {
  auto x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), binary_op, x_desc->data_type(), x_desc->data_type(),
      x_desc->shape().NumAxes());
}

template<typename T>
class PolygammaKernel final : public user_op ::OpKernel {
 public:
  PolygammaKernel() = default;
  ~PolygammaKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto n = ctx->Attr<int32_t>("n");

    const auto one = T{1};
    T temp = ((n % 2) ? one : -one) * exp(lgamma(static_cast<T>(n + one)));

    auto zeta = NewPrimitive(ctx, ep::primitive::BinaryOp::kZeta);

    CHECK(zeta);
    zeta->Launch(ctx->stream(), static_cast<T>(n) + one, x->shape_view().NumAxes(),
                 x->shape_view().ptr(), x->dptr(), out->mut_dptr());

    auto mul = NewPrimitive(ctx, ep::primitive::BinaryOp::kMul);
    CHECK(mul);
    mul->Launch(ctx->stream(), temp, out->shape_view().NumAxes(), out->shape_view().ptr(),
                out->dptr(), out->mut_dptr());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<ep::primitive::BinaryOp binary_op>
auto PrimitiveExists() {
  return hob::make_custom("PrimitiveExists", [](const user_op::KernelRegContext& ctx) -> bool {
    return NewPrimitive(&ctx, binary_op).operator bool();
  });
}
#define REGISTER_POLYGAMMA_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("polygamma")                                        \
      .SetCreateFn<PolygammaKernel<dtype>>()                               \
      .SetIsMatchedHob(PrimitiveExists<ep::primitive::BinaryOp::kZeta>()   \
                       && PrimitiveExists<ep::primitive::BinaryOp::kMul>() \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_POLYGAMMA_KERNEL(float);
REGISTER_POLYGAMMA_KERNEL(double)
}  // namespace
}  // namespace oneflow
