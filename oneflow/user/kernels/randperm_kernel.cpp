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
#include "oneflow/user/kernels/randperm_kernel.h"

namespace oneflow {

template<typename T>
class CpuRandPermKernel final : public user_op::OpKernel {
 public:
  CpuRandPermKernel() = default;
  ~CpuRandPermKernel() = default;
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeAutoGenerator());
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<RandpermKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* output = out->mut_dptr<T>();
    const int32_t N = ctx->Attr<int32_t>("N");
    auto* randperm_kernel_state = dynamic_cast<RandpermKernelState*>(state);
    CHECK_NOTNULL(randperm_kernel_state);
    const auto& generator = randperm_kernel_state->generator();
    const auto& cpu_generator = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
    CHECK_NOTNULL(generator);
    user_op::RangeFunctor<DeviceType::kCPU, T>()(ctx->device_ctx(), 0, 1, N, output);
    std::shuffle(output, output + N, cpu_generator->engine());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_CPU_RANDPERM_KERNEL(dtype)               \
  REGISTER_USER_KERNEL("randperm")                        \
      .SetCreateFn<CpuRandPermKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_RANDPERM_KERNEL(float)
REGISTER_CPU_RANDPERM_KERNEL(double)
REGISTER_CPU_RANDPERM_KERNEL(int32_t)
REGISTER_CPU_RANDPERM_KERNEL(int64_t)

}  // namespace oneflow
