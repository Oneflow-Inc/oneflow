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
#include "oneflow/core/operator/operator_util.h"
#include "oneflow/user/utils/pool_util.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

class AdaptivePool2DNpuKernel final : public user_op::OpKernel {
 public:
  AdaptivePool2DNpuKernel() = default;
  ~AdaptivePool2DNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { 
        user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
        user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
        std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
        NpuCommand npu_command;
        npu_command.OpName("AdaptiveAvgPool2d")
                   .Input(in_tensor)
                   .Output(out_tensor)
                   .Attr("output_size", output_size)
                   .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                   .Check();
        npu_command.Run()
                   .Realease();
    }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("adaptive_avg_pool2d")                                           \
    .SetCreateFn<AdaptivePool2DNpuKernel>()                            \
    .SetIsMatchedHob((user_op::HobDeviceType() == kNPU)); 

class AdaptivePool2DNpuGradKernel final : public user_op::OpKernel {
 public:
  AdaptivePool2DNpuGradKernel() = default;
  ~AdaptivePool2DNpuGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { 
        user_op::Tensor* grad_input = ctx->Tensor4ArgNameAndIndex("dx", 0);
        user_op::Tensor* grad_output = ctx->Tensor4ArgNameAndIndex("dy", 0);
        std::vector<int> dx_shape;
        for(size_t i=0; i<grad_input->shape().NumAxes();++i)
        {
            dx_shape.push_back(grad_input->shape().ptr()[i]);
        }  
        NpuCommand npu_command;
        npu_command.OpName("AdaptiveAvgPool2dGrad")
                   .Input(grad_output)
                   .Output(grad_input)
                   .Attr("orig_input_shape", dx_shape)
                   .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                   .Check();
        npu_command.Run()
                   .Realease();

    }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("adaptive_avg_pool2d_grad")                         \
    .SetCreateFn<AdaptivePool2DNpuGradKernel>()                          \
    .SetIsMatchedHob((user_op::HobDeviceType() == kNPU)); 

}