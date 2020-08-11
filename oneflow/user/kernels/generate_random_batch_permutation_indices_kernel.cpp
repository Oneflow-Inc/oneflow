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
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

class GenerateRandomBatchPermutationIndicesCPUKernel final : public user_op::OpKernel {
 public:
  GenerateRandomBatchPermutationIndicesCPUKernel() = default;
  ~GenerateRandomBatchPermutationIndicesCPUKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int64_t seed = ctx->Attr<int64_t>("seed");
    return std::make_shared<OpKernelStateWrapper<std::mt19937>>(seed);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* random_generator = dynamic_cast<OpKernelStateWrapper<std::mt19937>*>(state);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    std::iota(y->mut_dptr<int32_t>(), y->mut_dptr<int32_t>() + y->shape().elem_cnt(), 0);
    std::shuffle(y->mut_dptr<int32_t>(), y->mut_dptr<int32_t>() + y->shape().elem_cnt(),
                 *random_generator->Mutable());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("generate_random_batch_permutation_indices")
    .SetCreateFn<GenerateRandomBatchPermutationIndicesCPUKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "cpu");

}  // namespace oneflow
