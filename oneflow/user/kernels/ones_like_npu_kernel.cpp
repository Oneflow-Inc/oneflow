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
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace user_op {

namespace {


class OnesLikeNpuKernel final : public user_op::OpKernel {
 public:
  OnesLikeNpuKernel() = default;
  ~OnesLikeNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    NpuCommand npu_command;
    npu_command.OpName("OnesLike")
               .Input(in,"channel_nd")
               .Output(out,"channel_nd")
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    PrintResult(out);
    std::cout<<"Execute Over"<<std::endl; 
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


REGISTER_USER_KERNEL("ones_like")
    .SetCreateFn<OnesLikeNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);

}  // namespace

}  // namespace user_op

}  // namespace oneflow
