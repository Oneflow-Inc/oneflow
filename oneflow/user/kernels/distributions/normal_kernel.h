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

#ifndef ONEFLOW_USER_KERNELS_DISTRIBUTIONS_NORMAL_KERNEL_H_
#define ONEFLOW_USER_KERNELS_DISTRIBUTIONS_NORMAL_KERNEL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/normal_distribution.h"

namespace oneflow {

class NormalKernelState : public user_op::OpKernelState {
 public:
  explicit NormalKernelState(const std::shared_ptr<one::Generator>& generator)
      : generator_(generator) {}

  const std::shared_ptr<one::Generator>& generator() const { return generator_; }

 private:
  std::shared_ptr<one::Generator> generator_;
};

namespace {

template<DeviceType device_type, typename T>
class NormalKernel final : public user_op::OpKernel {
 public:
  NormalKernel() = default;
  ~NormalKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeAutoGenerator());
    generator->set_current_seed(ctx->Attr<int64_t>("seed"));
    return std::make_shared<NormalKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const double mean = ctx->Attr<double>("mean");
    const double std = ctx->Attr<double>("std");
    int64_t elem_cnt = out->shape().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();
    auto* normal_state = dynamic_cast<NormalKernelState*>(state);
    CHECK_NOTNULL(normal_state);
    const auto& generator = normal_state->generator();
    CHECK_NOTNULL(generator);
    NormalDistribution<device_type, T> distribution(static_cast<T>(mean), static_cast<T>(std));
    distribution(ctx->device_ctx(), elem_cnt, out_dptr, generator);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DISTRIBUTIONS_NORMAL_KERNEL_H_