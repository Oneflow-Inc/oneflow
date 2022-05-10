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
#include "oneflow/user/ops/npu_command.h"
#include "oneflow/core/ep/include/primitive/elementwise_unary.h"

namespace oneflow {

class ReluNpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluNpuKernel);
  ReluNpuKernel() = default;
  ~ReluNpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {

    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("x", 0);
    const user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("y", 0);    
    const int64_t elem_cnt = x->shape().elem_cnt();

    if (elem_cnt != 0) {
      NpuCommand npu_command;
      npu_command.OpName("Relu")
                 .Input(x, "channel_nd")
                 .Output(y, "channel_nd")
                 //.InputDesc(x_desc, "channel_nd")
                 //.OutputDesc(y_desc, "channel_nd")
                 .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                 .Check();
      npu_command.Run();
    OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    PrintResult(y);
    std::cout<<"Execute Over"<<std::endl;       
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("relu").SetCreateFn<ReluNpuKernel>().SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);

}  // namespace oneflow
