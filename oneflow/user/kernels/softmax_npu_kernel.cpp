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
#include "oneflow/core/ep/include/primitive/softmax.h"
#include "oneflow/core/ep/include/primitive/softmax_backward.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

class SoftmaxNpuKernel final : public user_op::OpKernel {
 public:
  SoftmaxNpuKernel() = default;
  ~SoftmaxNpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape_view();
    NpuCommand npu_command;
    npu_command.OpName("SoftmaxV2")
                .Input(in)
                .Attr("axes", (int64_t)-1)
                .Output(out)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
    npu_command.Run()
               .Realease();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("softmax").SetCreateFn<SoftmaxNpuKernel>().SetIsMatchedHob(
    user_op::HobDeviceType() == DeviceType::kNPU);

class SoftmaxGradNpuKernel final : public user_op::OpKernel {
 public:
  SoftmaxGradNpuKernel() = default;
  ~SoftmaxGradNpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    NpuCommand npu_command;
    npu_command.OpName("SoftmaxGrad")
                .Input(y)
                .Input(dy)
                .Output(dx)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
    npu_command.Run()
               .Realease();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("softmax_grad")
    .SetCreateFn<SoftmaxGradNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);
} // namespace oneflow