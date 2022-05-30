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
#include "oneflow/core/ep/include/primitive/log_softmax.h"
#include "oneflow/core/ep/include/primitive/log_softmax_backward.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace {
class LogSoftmaxNpuKernel final : public user_op::OpKernel{
 public:
  LogSoftmaxNpuKernel() = default;
  ~LogSoftmaxNpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    NpuCommand npu_command;
    std::vector<int64_t> axes = {-1};
    npu_command.OpName("LogSoftmaxV2")
            .Input(in)
            .Output(prob)
            .Attr("axes",axes)
            .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
            .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));  
    //PrintResult(prob);
    //std::cout<<"LogSoftmaxNpuKernel Execute Over"<<std::endl;  
      
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class LogSoftmaxGradNpuKernel final : public user_op::OpKernel {
 public:
  LogSoftmaxGradNpuKernel() = default;
  ~LogSoftmaxGradNpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    std::vector<int64_t> axes = {-1};
    NpuCommand npu_command;
    npu_command.OpName("LogSoftmaxGrad")
            .Input(dy)
            .Input(prob)
            .Attr("axes",axes)
            .Output(dx)
            .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
            .Check();
    npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));  
    //PrintResult(dx);
    //std::cout<<"LogSoftmaxNpuGradKernel Execute Over"<<std::endl;  

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

REGISTER_USER_KERNEL("log_softmax")
    .SetCreateFn<LogSoftmaxNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);

REGISTER_USER_KERNEL("log_softmax_grad")
    .SetCreateFn<LogSoftmaxGradNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);

}  // namespace oneflow
