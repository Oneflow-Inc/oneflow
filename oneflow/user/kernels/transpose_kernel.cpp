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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/common/primitive/permute.h"

namespace oneflow {

namespace user_op {

namespace {

bool IsIdentity(const ShapeView& in_shape, const std::vector<int32_t>& perm) {
  constexpr int kMaxNumDims = 12;
  CHECK_LE(in_shape.NumAxes(), kMaxNumDims);
  CHECK_EQ(in_shape.NumAxes(), perm.size());

  size_t simplified_num_dims{};
  int64_t simplified_src_dims[kMaxNumDims]{};
  int simplified_permutation[kMaxNumDims]{};
  ep::primitive::permute::SimplifyPermutation<kMaxNumDims>(
      in_shape.NumAxes(), in_shape.ptr(), perm.data(), &simplified_num_dims, simplified_src_dims,
      simplified_permutation);
  for (int i = 0; i < simplified_num_dims; ++i) {
    if (simplified_permutation[i] != i) { return false; }
  }
  return true;
}

}  // namespace

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx) {
  const int64_t num_dims = ctx->TensorDesc4ArgNameAndIndex("output", 0)->shape().NumAxes();
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

class TransposeKernel final : public OpKernel, public user_op::CudaGraphSupport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TransposeKernel);
  TransposeKernel() = default;
  ~TransposeKernel() override = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    auto primitive = NewPermutePrimitive(ctx);
    CHECK(primitive);

    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("input", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto& perm = ctx->Attr<std::vector<int32_t>>("perm");
    const ShapeView& in_shape = tensor_in->shape_view();
    DataType dtype = tensor_out->data_type();
    size_t num_dims = tensor_in->shape_view().NumAxes();
    const int64_t* src_dims = in_shape.ptr();

    int64_t elem_cnt = tensor_out->shape_view().elem_cnt();

    if (elem_cnt != 0) {
      if (IsIdentity(in_shape, perm)) {
        // if permute vector is 0,1,...,n, do data copy directly
        AutoMemcpy(ctx->stream(), tensor_out->mut_dptr(), tensor_in->dptr(),
                   elem_cnt * GetSizeOfDataType(dtype), tensor_out->mem_case(),
                   tensor_in->mem_case());
      } else {
        primitive->Launch(ctx->stream(), dtype, num_dims, src_dims, tensor_in->dptr(), perm.data(),
                          tensor_out->mut_dptr());
      }

    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

auto PermutePrimitiveExists() {
  return hob::make_custom("PermutePrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewPermutePrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("transpose")
    .SetCreateFn<TransposeKernel>()
    .SetIsMatchedHob(PermutePrimitiveExists() == true)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      const ShapeView input_shape(ctx.InputShape("input", 0));
      const auto& perm = ctx.Attr<std::vector<int32_t>>("perm");
      if (IsIdentity(input_shape, perm)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("output", 0, "input", 0, false));
      }
      return Maybe<void>::Ok();
    });

}  // namespace user_op
}  // namespace oneflow
