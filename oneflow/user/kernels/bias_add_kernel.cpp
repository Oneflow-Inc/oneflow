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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::BroadcastElementwiseBinary> NewPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("a", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
      ctx->device_type(), ep::primitive::BinaryOp::kAdd, data_type, data_type, 3);
}

class BiasAddUserKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  BiasAddUserKernel() = default;
  ~BiasAddUserKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    if (a_tensor->shape_view().elem_cnt() == 0 || b_tensor->shape_view().elem_cnt() == 0) {
      return;
    }
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const int64_t outer_size = a_tensor->shape_view().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape_view().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape_view().Count(bias_add_axis + 1);
    auto primitive = NewPrimitive(ctx);
    const int64_t src0_dims[3] = {outer_size, bias_size, inner_size};
    const int64_t src1_dims[3] = {1, bias_size, 1};
    primitive->Launch(ctx->stream(), 3, src0_dims, a_tensor->dptr(), 3, src1_dims, b_tensor->dptr(),
                      out_tensor->mut_dptr());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto PrimitiveExists() {
  return hob::make_custom("PrimitiveExists", [](const user_op::KernelRegContext& ctx) -> bool {
    return NewPrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("bias_add")
    .SetCreateFn<BiasAddUserKernel>()
    .SetIsMatchedHob(PrimitiveExists() == true)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "a", 0, true));
      return Maybe<void>::Ok();
    });
}  // namespace

}  // namespace oneflow
