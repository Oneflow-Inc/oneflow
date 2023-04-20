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
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/distributions/normal_distribution.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {
template<DeviceType device_type, typename T>
class CpuNormalTensorFloatKernel final : public user_op::OpKernel {
 public:
  CpuNormalTensorFloatKernel() = default;
  ~CpuNormalTensorFloatKernel() = default;
 
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(device_type));
    // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const double std = ctx->Attr<double>("std");
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = mean->shape_view().elem_cnt();
    
    const T* mean_ptr = mean->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    auto* distribution_state = dynamic_cast<DistributionKernelState*>(state);
    CHECK_NOTNULL(distribution_state);
    const auto& generator = distribution_state->generator();
    CHECK_NOTNULL(generator);
    
    auto gen = CHECK_JUST(generator->Get<ep::CPUGenerator>());
    CHECK_GE(elem_cnt, 0) << "elem_cnt must be non-negative, but got " << elem_cnt;

    std::normal_distribution<T> random_distribution(0, std);
    //  mean + output * std
    FOR_RANGE(int32_t, i, 0, elem_cnt) { out_ptr[i] = mean[i] + std * random_distribution(gen->engine()); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_NORMAL_TENSOR_FLOAT_KERNEL(device,dtype)                              \
  REGISTER_USER_KERNEL("normal_tensor_float")                                       \
      .SetCreateFn<CpuNormalTensorFloatKernel<device,dtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)               \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_NORMAL_TENSOR_FLOAT_KERNEL(DeviceType::kCPU, float16)
REGISTER_CPU_NORMAL_TENSOR_FLOAT_KERNEL(DeviceType::kCPU, float)
REGISTER_CPU_NORMAL_TENSOR_FLOAT_KERNEL(DeviceType::kCPU, double)

}  // namespace oneflow
