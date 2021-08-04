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

#include "oneflow/user/kernels/randint_kernel.h"

namespace oneflow {

class CpuRandintKernel final : public user_op::OpKernel {
 public:
  CpuRandintKernel() = default;
  ~CpuRandintKernel() = default;
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeAutoGenerator());
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<RandintKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t* output = out->mut_dptr<int64_t>();
    auto* randint_kernel_state = dynamic_cast<RandintKernelState*>(state);
    CHECK_NOTNULL(randint_kernel_state);
    const auto& generator = randint_kernel_state->generator();
    const auto& cpu_generator = CHECK_JUST(generator->Get<one::CPUGeneratorImpl>());
    CHECK_NOTNULL(generator);
    const int64_t n = out->shape().elem_cnt();
    const int64_t low = ctx->Attr<int64_t>("low");
    const int64_t high = ctx->Attr<int64_t>("high");
    std::uniform_int_distribution<int64_t> dis(low, high - 1);
    XPU_1D_KERNEL_LOOP(i, n)
    output[i] = dis(cpu_generator->engine());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("randint").SetCreateFn<CpuRandintKernel>().SetIsMatchedHob( 
      (user_op::HobDeviceTag() == "cpu"));


}  // namespace oneflow
