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
#ifndef ONEFLOW_USER_KERNELS_DISTRIBUTIONS_UNIFORM_INT_KERNEL_H_
#define ONEFLOW_USER_KERNELS_DISTRIBUTIONS_UNIFORM_INT_KERNEL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/distributions/uniform_int_distribution.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

namespace {

// The following algorithm is adopted from pytorch:
// The purpose of `update_from` and `update_to` is to find the closest valid int64_t number that can
// be used as actual `from`. The current implementation of `random_` uses uint64_t arithmetics and
// casts the result to the target dtype(scalar_t). This casting can result in generating numbers
// that happen to be greater or equal to `to` value. For instance:
//
//    auto actual = torch::empty({3, 3}, torch::half);
//    actual.random_(0, 65504);
//
// If random's uint64_t arithmetics produces 65503 as a random value after casting to torch::half it
// becomes 65504 and violates the requirement that random value must be less than `to`. To resolve
// this issue `update_from` and `update_to` moves `from` to the right and `to` to the left to the
// next closest value that won't go outside [from, to) after casting to the target dtype. For `to` =
// 65504 it moves left for (1 << (log2(to) - 11 + 1)) = 32 and becomes 65472, which is previous
// available number for torch::half dtype.
template<typename scalar_t>
int64_t update_from(int64_t from) {
  const auto from_plus_1 = static_cast<int64_t>(static_cast<scalar_t>(from + 1));
  if (from_plus_1 < from) {
    int64_t from_ = std::abs(from + 1);
    int n = 0;
    while (from_ >>= 1) ++n;
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    from = from_plus_1 + (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return from;
}

template<typename scalar_t>
int64_t update_to(int64_t to) {
  const auto to_minus_1 = static_cast<int64_t>(static_cast<scalar_t>(to - 1));
  if (to_minus_1 >= to) {
    int64_t to_ = std::abs(to - 1);
    int n = 0;
    while (to_ >>= 1) ++n;
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    to = to_minus_1 - (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
  }
  return to;
}

template<DeviceType device_type, typename T>
class UniformIntKernel final : public user_op::OpKernel {
 public:
  UniformIntKernel() = default;
  ~UniformIntKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeAutoGenerator());
    // When SBP is Spit, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t from = ctx->Attr<int64_t>("from");
    int64_t to = ctx->Attr<int64_t>("to");
    CHECK_LT(from, to) << "uniform kernel expects 'from' to be less than 'to'";

    if (IsFloating<T>::value) {
      from = update_from<T>(from);
      to = update_to<T>(to);
      CHECK_LT(from, to) << "uniform kernel expects 'from' casted to dtype to be less than 'to'"
                            " casted to dtype";
    }
    check_from_to_in_range<T>(from, to - 1);
    int64_t elem_cnt = out->shape_view().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    UniformIntDistribution<device_type, T> distribution(from, to);
    distribution(ctx->stream(), elem_cnt, out_dptr, generator);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DISTRIBUTIONS_UNIFORM_INT_KERNEL_H_
