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
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/batch_matmul.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace {

class MatmulNpuKernel final : public user_op::OpKernel {
 public:
  MatmulNpuKernel() = default;
  ~MatmulNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    bool trans_a = ctx->Attr<bool>("transpose_a");
    bool trans_b = ctx->Attr<bool>("transpose_b");
    user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    CHECK_EQ(a->shape().NumAxes(), 2);
    const DataType data_type = a->data_type();
    user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(b->shape().NumAxes(), 2);
    CHECK_EQ(b->data_type(), data_type);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out->shape().NumAxes(), 2);
    CHECK_EQ(out->data_type(), data_type);
    int64_t offset_x = 0;
    NpuCommand npu_command;
    npu_command.OpName("MatMulV2")
               .Input(a, "channels_nd")
               .Input(b, "channels_nd")
               .Output(out, "channels_nd")
               .Attr("transpose_x1", trans_a)
               .Attr("transpose_x2", trans_b)
               .Attr("offset_x", offset_x)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(out);
    //std::cout<<"Matmul Execute Over"<<std::endl; 
  }
};

REGISTER_USER_KERNEL("matmul")
    .SetCreateFn<MatmulNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);
    // dck_caution_here
    // .SetInplaceProposalFn([](const user_op::InferContext& ctx,
    //                          const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
    //   if (ctx.has_input("_add_to_output", 0)) {
    //     OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true));
    //   }
    //   return Maybe<void>::Ok();
    // });    
} // namespace {anonymous}
} // namespace oneflow