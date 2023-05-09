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

template<typename T>
class PolygammaKernel final : public user_op : OpKernel {
 public:
  PolygammaKernel() = default;
  ~PolygammaKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto n = ctx->Attr<int32_t>("n");

    const auto one = T{1};
    T temp = ((n % 2) ? one : -one) * exp(lgamma(static_assert<T>(n) + one));

    std::unique_ptr<ep::primitive::BinaryOp> zeta =
        ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
            ctx->device_type(), ep::primitive::UnaryOp::kZeta, x->data_type(), x->data_type());
    CHECK(zeta);
    zeta->Launch(ctx->stream(), static_assert<T>(n) + one, x->shape_view().NumAxes(),
                 x->shape_view(), x->dptr(), out->mut_dptr());

    std::unique_ptr<ep::primitive::BinaryOp> mul =
        ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
            ctx->device_type(), ep::primitive::UnaryOp::kMul, x->data_type(), x->data_type());
    CHECK(mul);
    mul->Launch(ctx->stream(), temp, out->shape_view().NumAxes(), out->shape_view(), out->dptr(),
                out->mut_dptr());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto PrimitiveExists() {
  return hob::make_custom("MathBinaryBroadcastPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) -> bool {
                            return NewPrimitive(&ctx).operator bool();
                          });
}

#define REGISTER_POLYGAMMA_KERNEL(dtype)     \
  REGISTER_USER_KERNEL("polygamma")          \
      .SetCreateFn<PolygammaKernel<dtype>>() \
      .SetIsMatchedHob(PrimitiveExists()     \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_POLYGAMMA_KERNEL(float);
REGISTER_POLYGAMMA_KERNEL(double)
}  // namespace

}  // namespace oneflow