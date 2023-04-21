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
#include "oneflow/user/kernels/normal_with_tensor_kernel_util.h"

namespace oneflow {
/*
The CpuNormalFloatTensorKernel class is an implementation of the OpKernel class.
It is used to compute the normal distribution of a float tensor on the CPU.
*/
template<DeviceType device_type, typename T>
class CpuNormalFloatTensorKernel final : public user_op::OpKernel {
 public:
  CpuNormalFloatTensorKernel() = default;
  ~CpuNormalFloatTensorKernel() = default;

  /*
  CreateOpKernelState creates a new OpKernelState object and sets the current seed for the
  generator.
  */
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(device_type));
    // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }

 private:
  /*
  Compute computes the normal distribution of a float tensor.
  */
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const double mean = ctx->Attr<double>("mean");
    const user_op::Tensor* std = ctx->Tensor4ArgNameAndIndex("std", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    normal_out_impl<device_type, T>(ctx, state, out, mean, std);
    //  mean + output * std
    int64_t elem_cnt = out->shape_view().elem_cnt();
    T* out_dptr = out->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { out_dptr[i] = mean + std[i] * out_dptr[i]; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_NORMAL_FLOAT_TENSOR_KERNEL(device, dtype)  \
  REGISTER_USER_KERNEL("normal_float_tensor")                   \
      .SetCreateFn<CpuNormalFloatTensorKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)     \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_NORMAL_FLOAT_TENSOR_KERNEL(DeviceType::kCPU, float16)
REGISTER_CPU_NORMAL_FLOAT_TENSOR_KERNEL(DeviceType::kCPU, float)
REGISTER_CPU_NORMAL_FLOAT_TENSOR_KERNEL(DeviceType::kCPU, double)

}  // namespace oneflow
