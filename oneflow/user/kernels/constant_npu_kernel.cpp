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
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {
namespace user_op {

namespace {

class ConstantNpuKernel final : public OpKernel {
 public:
  ConstantNpuKernel() = default;
  ~ConstantNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool is_floating_value = ctx->Attr<bool>("is_floating_value");
    const int64_t elem_cnt = out_tensor->shape().elem_cnt();
    const int64_t len = elem_cnt * GetSizeOfDataType(out_tensor->data_type());
    CHECK_GE(elem_cnt, 0);
    if (elem_cnt == 0) { return; }
    //OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));
    if(is_floating_value)
    {
        float value = ctx->Attr<double>("floating_value");
        NpuCommand npu_command;
        npu_command.OpName("Fills")
                .Input(out_tensor,"channels_nd")
                .Output(out_tensor,"channels_nd")
                .Attr("value",value)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
        npu_command.Run()
                .Realease();   
    }
    else
    {
        float value = ctx->Attr<int64_t>("integer_value");
        NpuCommand npu_command;
        npu_command.OpName("Fills")
                .Input(out_tensor,"channels_nd")
                .Output(out_tensor,"channels_nd")
                .Attr("value",value)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
        npu_command.Run()
                .Realease();   
    }
    
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("constant")
    .SetCreateFn<ConstantNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);

}  // namespace

}  // namespace user_op
}  // namespace oneflow
