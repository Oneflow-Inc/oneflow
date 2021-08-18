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
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/user/kernels/range_kernel_util.h"
#include "oneflow/user/kernels/distributions/uniform_kernel.h"
namespace oneflow {

class CpuRandPermKernel final : public user_op::OpKernel {
 public:
  CpuRandPermKernel() = default;
  ~CpuRandPermKernel() = default;
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeAutoGenerator());
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<UniformKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int32_t* output = out->mut_dptr<int32_t>();
    const int32_t n = ctx->Attr<int32_t>("n");
    auto* randperm_kernel_state = dynamic_cast<UniformKernelState*>(state);
    CHECK_NOTNULL(randperm_kernel_state);
    const auto& generator = randperm_kernel_state->generator();
    const auto& cpu_generator = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
    CHECK_NOTNULL(generator);
    user_op::RangeFunctor<DeviceType::kCPU, int32_t>()(ctx->device_ctx(), 0, 1, n, output);
    std::shuffle(output, output + n, cpu_generator->engine());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("randperm")
    .SetCreateFn<CpuRandPermKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu"));

}  // namespace oneflow
