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
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/tensor_meta.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_unary.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseUnary> NewBroadcastPrimitive(Context* ctx) {
  const DataType in_data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  const DataType out_data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  const size_t max_ndim = std::max(ctx->TensorDesc4ArgNameAndIndex("in", 0)->shape().NumAxes(),
                                   ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape().NumAxes());
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseUnaryFactory>(
      ctx->device_type(), ep::primitive::UnaryOp::kCast, in_data_type, out_data_type, max_ndim);
}

class CastKernel final : public OpKernel, public user_op::CudaGraphSupport {
 public:
  CastKernel() = default;
  ~CastKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* output = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = input->shape_view().elem_cnt();
    // 0-size tensor
    CHECK_EQ(output->shape_view().elem_cnt(), elem_cnt)
        << "The number of cast op's input and output elements should be equal.";
    if (elem_cnt == 0) { return; }
    if (input->data_type() == output->data_type() && input->dptr() == output->dptr()) { return; }
    const size_t ndim = input->shape_view().NumAxes();
    auto broadcast_primitive = NewBroadcastPrimitive(ctx);
    CHECK(broadcast_primitive);
    if (ndim == 0 && elem_cnt == 1) {
      // 0-dim tensor
      // TODO: remove these when BroadcastElementwiseUnary primitive support 0-dim(scalar) tensor
      Shape input_shape(DimVector{1});
      Shape output_shape(DimVector{1});
      Stride input_stride(DimVector{1});
      Stride output_stride(DimVector{1});
      const size_t scalar_ndim = 1;
      broadcast_primitive->Launch(ctx->stream(), scalar_ndim, input_shape.data(),
                                  input_stride.data(), input->dptr(), scalar_ndim,
                                  output_shape.data(), output_stride.data(), output->mut_dptr());
    } else {
      broadcast_primitive->Launch(
          ctx->stream(), ndim, input->shape_view().data(), input->stride().data(), input->dptr(),
          ndim, output->shape_view().data(), output->stride().data(), output->mut_dptr());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto BroadcastPrimitiveExists() {
  return hob::make_custom("BroadcastElementwiseUnaryPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) -> bool {
                            return NewBroadcastPrimitive(&ctx).operator bool();
                          });
}

REGISTER_USER_KERNEL("cast")
    .SetCreateFn<CastKernel>()
    .SetIsMatchedHob(BroadcastPrimitiveExists() == true)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.InputDType("in", 0) == ctx.Attr<DataType>("dtype")) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("cast_like")
    .SetCreateFn<CastKernel>()
    .SetIsMatchedHob(BroadcastPrimitiveExists() == true)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.InputDType("in", 0) == ctx.InputDType("like", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));
      }
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace user_op

}  // namespace oneflow
