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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"

namespace oneflow {

namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Cast> NewCastPrimitive(Context* ctx) {
  const DataType in_data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  const DataType out_data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::CastFactory>(ctx->device_type(), in_data_type,
                                                                 out_data_type);
}

class CastKernel final : public OpKernel, public user_op::CudaGraphSupport {
 public:
  CastKernel() = default;
  ~CastKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* output_tenor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = input_tensor->shape_view().elem_cnt();
    CHECK_EQ(output_tenor->shape_view().elem_cnt(), elem_cnt);
    if (input_tensor->data_type() == output_tenor->data_type()
        && input_tensor->dptr() == output_tenor->dptr()) {
      return;
    }
    auto primitive = NewCastPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), input_tensor->dptr(), output_tenor->mut_dptr(), elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto CastPrimitiveExists() {
  return hob::make_custom("CastPrimitiveExists", [](const user_op::KernelRegContext& ctx) -> bool {
    return NewCastPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("cast")
    .SetCreateFn<CastKernel>()
    .SetIsMatchedHob(CastPrimitiveExists() == true)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.InputDType("in", 0) == ctx.Attr<DataType>("dtype")) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("cast_like")
    .SetCreateFn<CastKernel>()
    .SetIsMatchedHob(CastPrimitiveExists() == true)
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
