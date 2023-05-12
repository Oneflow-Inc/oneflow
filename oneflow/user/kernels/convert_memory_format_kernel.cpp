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
#include "oneflow/core/common/memory_format_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/convert_memory_format_util.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/common/primitive/permute.h"

namespace oneflow {

class ConvertMemoryFormatKernel final : public user_op::OpKernel {
 public:
  ConvertMemoryFormatKernel() = default;
  ~ConvertMemoryFormatKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    ConvertMemoryFormat(ctx->stream(), in->shape_view(), in->data_type(), in->dptr(),
                        out->mut_dptr(), in->memory_format(), out->memory_format());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename Context>
std::unique_ptr<ep::primitive::Permute> NewPermutePrimitive(Context* ctx) {
  const int64_t num_dims = ctx->TensorDesc4ArgNameAndIndex("output", 0)->shape().NumAxes();
  return ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), num_dims);
}

auto PermutePrimitiveExists() {
  return hob::make_custom("PermutePrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewPermutePrimitive(&ctx).operator bool();
  });
}

REGISTER_USER_KERNEL("convert_memory_format")
    .SetCreateFn<ConvertMemoryFormatKernel>()
    .SetIsMatchedHob(PermutePrimitiveExists() == true);

}  // namespace oneflow
