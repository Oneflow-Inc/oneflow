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
#ifndef ONEFLOW_USER_KERNEL_RANDOM_MASK_GENERATOR_H_
#define ONEFLOW_USER_KERNEL_RANDOM_MASK_GENERATOR_H_
#include "oneflow/user/kernels/random_mask_generator.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

class RandomMaskLikeKernelState : public user_op::OpKernelState {
 public:
  explicit RandomMaskLikeKernelState(const std::shared_ptr<one::Generator>& generator)
      : generator_(generator) {}

  const std::shared_ptr<one::Generator> generator() { return generator_; }

 private:
  const std::shared_ptr<one::Generator> generator_;
};

namespace {

template<DeviceType device_type>
class RandomMaskLikeKernel final : public user_op::OpKernel {
 public:
  RandomMaskLikeKernel() = default;
  ~RandomMaskLikeKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& gen = one::Generator::GetDefaultGenerator();
    return std::make_shared<RandomMaskLikeKernelState>(gen);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* like = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t elem_cnt = like->shape().elem_cnt();
    int8_t* mask = out->mut_dptr<int8_t>();
    auto* rml_state = dynamic_cast<RandomMaskLikeKernelState*>(state);
    CHECK_NOTNULL(rml_state);
    auto generator = rml_state->generator();
    CHECK_NOTNULL(generator);
    auto rml_gen = std::make_shared<RandomMaskGenerator<device_type>>(generator);
    rml_gen->Generate(ctx->device_ctx(), elem_cnt, ctx->Attr<float>("rate"), mask);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNEL_RANDOM_MASK_GENERATOR_H_
