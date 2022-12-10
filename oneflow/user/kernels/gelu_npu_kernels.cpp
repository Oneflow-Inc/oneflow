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

class GeluNpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluNpuKernel);
  GeluNpuKernel() = default;
  ~GeluNpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {

    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::TensorDesc* x_desc = ctx->TensorDesc4ArgNameAndIndex("in", 0);
    const user_op::TensorDesc* y_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);    
    const int64_t elem_cnt = x->shape_view().elem_cnt();

    if (elem_cnt != 0) {
      NpuCommand npu_command;
      npu_command.OpName("Gelu")
                 .Input(x, "channels_nd")
                 .Output(y, "channels_nd")
                 .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                 .Check();
      npu_command.Run()
               .Realease();
    //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    //PrintResult(y);
    //std::cout<<"Gelu Execute Over"<<std::endl;       
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("gelu").SetCreateFn<GeluNpuKernel>().SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);


class GeluGradNpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluGradNpuKernel);
  GeluGradNpuKernel() = default;
  ~GeluGradNpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {

    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* unused = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t elem_cnt = x->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      NpuCommand npu_command;
      npu_command.OpName("GeluGrad")
                 .Input(dy, "channel_nd")
                 .Input(x, "channel_nd")
                 .Input(unused, "channel_nd")
                 .Output(dx, "channel_nd")
                 .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                 .Check();
      npu_command.Run()
               .Realease();
      //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
      //PrintResult(dx);
      //std::cout<<"Execute Over"<<std::endl;       
    } else {
      // For 0-d Tensor
      UNIMPLEMENTED();
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("gelu_grad").SetCreateFn<GeluGradNpuKernel>().SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);

}  // namespace oneflow
