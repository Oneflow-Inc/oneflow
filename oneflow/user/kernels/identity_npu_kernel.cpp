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
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace {

class IdentityNpuKernel final : public user_op::OpKernel {
 public:
  IdentityNpuKernel() = default;
  ~IdentityNpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape_view();
    CHECK_EQ(out->shape_view(), in_shape);
    const DataType in_data_type = in->data_type();
    CHECK_EQ(out->data_type(), in_data_type);
    std::cout<<"Identity"<<std::endl;
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream())); 
    // PrintResult(in);
    // Memcpy<DeviceType::kNPU>(ctx->stream(), out->mut_dptr<void>(), in->dptr<void>(),
    //                     in_shape.elem_cnt() * GetSizeOfDataType(in_data_type));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_IDENTITY_KERNEL(op)                                                    \
  REGISTER_USER_KERNEL(op)                                                                      \
      .SetCreateFn<IdentityNpuKernel>()                                                    \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU)                                      \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));                      \
        return Maybe<void>::Ok();                                                               \
      });
REGISTER_IDENTITY_KERNEL("identity")
} // namespace

} // namespace oneflow