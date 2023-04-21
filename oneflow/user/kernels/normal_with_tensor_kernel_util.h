

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

#ifndef ONEFLOW_USER_KERNELS_NORMAL_WITH_TENSOR_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_NORMAL_WITH_TENSOR_KERNEL_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/distributions/normal_distribution.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

namespace {

// template<DeviceType device_type, typename T>
// class CustomNormalWithTensorKernel  : public user_op::OpKernel {
//  public:
//   CustomNormalWithTensorKernel() = default;
//   ~CustomNormalWithTensorKernel() = default;

//   std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
//       user_op::KernelInitContext* ctx) const override {
//     const auto& generator = CHECK_JUST(one::MakeGenerator(device_type));
//     // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
//     generator->set_current_seed(
//         CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
//     return std::make_shared<DistributionKernelState>(generator);
//   }

// };

template<DeviceType device_type, typename T>
void normal_out_impl(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                      user_op::Tensor* out,  const double mean, const user_op::Tensor* std)  {
    // TODO(fengwen) std all elements of std >= 0.0
    int64_t elem_cnt = out->shape_view().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    NormalDistribution<device_type, T> distribution(static_cast<T>(0), static_cast<T>(1));
    distribution(ctx->stream(), elem_cnt, out_dptr, generator);
}

template<DeviceType device_type, typename T>
void normal_out_impl(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                       user_op::Tensor* out, const user_op::Tensor* mean, const double std)  {
    CHECK_GE(std, static_cast<T>(0.0));
    int64_t elem_cnt = out->shape_view().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    NormalDistribution<device_type, T> distribution(static_cast<T>(0.0), static_cast<T>(std));
    distribution(ctx->stream(), elem_cnt, out_dptr, generator);
}

template<DeviceType device_type, typename T>
void normal_out_impl(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                      user_op::Tensor* out, const user_op::Tensor* mean, const user_op::Tensor* std) {
    // TODO(fengwen) std all elements of std >= 0.0
    int64_t elem_cnt = out->shape_view().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    NormalDistribution<device_type, T> distribution(static_cast<T>(0), static_cast<T>(1));
    distribution(ctx->stream(), elem_cnt, out_dptr, generator);
}

template<DeviceType device_type, typename T>
void normal_out_impl(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                      user_op::Tensor* out, const double mean, const double std) {
    CHECK_GE(std, static_cast<T>(0.0));
    int64_t elem_cnt = out->shape_view().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    NormalDistribution<device_type, T> distribution(static_cast<T>(mean), static_cast<T>(std));
    distribution(ctx->stream(), elem_cnt, out_dptr, generator);
}

}  // namespace
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_NORMAL_WITH_TENSOR_KERNEL_UTIL_H_
