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
#include "oneflow/user/kernels/scalar_math_kernels.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {


template<typename T>
class ScalarMulNpuKernel final : public user_op::OpKernel {
 public:
  ScalarMulNpuKernel() = default;
  ~ScalarMulNpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    AclTensorWrapper wrap(nullptr, ACL_FLOAT, 0, nullptr,
                             ACL_FORMAT_ND, sizeof(T), &scalar_operand);//dck_caution_here ACL_FLOAT/ACL_FLOAT16
    NpuCommand npu_command;
    npu_command.OpName("Mul")
                .Input(in)
                .Input(wrap)
                .Output(out)
                .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
                .Check();
    npu_command.Run();
    PrintResult(out);
    std::cout<<"ScalarMulNpuKernel Execute Over"<<std::endl; 

  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_SCALAR_MUL_NPU_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("scalar_mul")                                               \
      .SetCreateFn<ScalarMulNpuKernel<dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU)                                \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));
REGISTER_SCALAR_MUL_NPU_KERNEL(float);
REGISTER_SCALAR_MUL_NPU_KERNEL(float16);
} // namespace oneflow