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
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/user/kernels/op_kernel_wrapper.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace user_op {

namespace {


class CastNpuKernel final : public OpKernel{
 public:
  CastNpuKernel() = default;
  ~CastNpuKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    Tensor* input_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = input_tensor->shape().elem_cnt();
    CHECK_EQ(output_tensor->shape().elem_cnt(), elem_cnt);
    if (input_tensor->data_type() == output_tensor->data_type()
        && input_tensor->dptr() == output_tensor->dptr()) {
      return;
    }
    NpuCommand npu_command;
    npu_command.OpName("Cast")
               .Input(input_tensor, "channels_nd")
               .Output(output_tensor, "channels_nd")
               .Attr("dst_type", (int64_t)dataTypeMap(output_tensor->data_type()))
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(output_tensor);
    //std::cout<<"Cast Execute Over"<<std::endl;       
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


REGISTER_USER_KERNEL("cast")
    .SetCreateFn<CastNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);
    // .SetInplaceProposalFn([](const user_op::InferContext& ctx,
    //                          const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
    //   if (ctx.InputDType("in", 0) == ctx.Attr<DataType>("dtype")) {
    //     OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));
    //   }
    //   return Maybe<void>::Ok();
    // });

// REGISTER_USER_KERNEL("cast_like")
//     .SetCreateFn<CastKernel>()
//     .SetIsMatchedHob(CastPrimitiveExists() == true)
//     .SetInplaceProposalFn([](const user_op::InferContext& ctx,
//                              const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
//       if (ctx.InputDType("in", 0) == ctx.InputDType("like", 0)) {
//         OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));
//       }
//       return Maybe<void>::Ok();
//     });

}  // namespace

}  // namespace user_op

}  // namespace oneflow
