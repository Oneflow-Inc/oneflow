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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_unary.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseUnary> NewPrimitive(Context* ctx) {
  const auto* in_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
  const auto* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);
  size_t max_ndim = std::max(in_desc->shape().size(), out_desc->shape().size());
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseUnaryFactory>(
      ctx->device_type(), ep::primitive::UnaryOp::kIdentity, in_desc->data_type(),
      out_desc->data_type(), max_ndim);
}

auto PrimitiveExists() {
  return hob::make_custom("BroadcastElementwiseUnaryPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) -> bool {
                            return NewPrimitive(&ctx).operator bool();
                          });
}

}  // namespace

class ExpandKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  ExpandKernel() = default;
  ~ExpandKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto in_shape = in->shape_view();
    auto out_shape = out->shape_view();

    // handle 0-size tensor
    if (std::any_of(out_shape.begin(), out_shape.end(), [](int64_t dim) { return dim <= 0; })) {
      return;
    }

    auto prim = NewPrimitive(ctx);
    CHECK(prim);
    if (in_shape.size() == 0 && in_shape.elem_cnt() == 1) {
      // handle 0-dim tensor
      // NOTE: this handle will be remove when BroadcastElementwiseUnary primitive support 0-dim
      // tensor
      int64_t scalar_ndim = 1;
      Shape scalar_shape(DimVector{scalar_ndim});
      Shape scalar_stride(DimVector{scalar_ndim});
      prim->Launch(ctx->stream(), scalar_ndim, scalar_shape.data(), scalar_stride.data(),
                   in->dptr(), out_shape.size(), out_shape.data(), out->stride().data(),
                   out->mut_dptr());
    } else {
      prim->Launch(ctx->stream(), in_shape.size(), in_shape.data(), in->stride().data(), in->dptr(),
                   out_shape.size(), out_shape.data(), out->stride().data(), out->mut_dptr());
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("expand").SetCreateFn<ExpandKernel>().SetIsMatchedHob(PrimitiveExists()
                                                                           == true);

}  // namespace oneflow
