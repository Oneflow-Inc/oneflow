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
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/distributions/normal_distribution.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class CustomNormalWithTensorKernel : public user_op::OpKernel {
 public:
  CustomNormalWithTensorKernel() = default;
  ~CustomNormalWithTensorKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(device_type));
    // When SBP is Split, each rank uses a different seeds, otherwise, ranks use the same seed
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<DistributionKernelState>(generator);
  }
};

template<typename T, typename ComputeType>
struct CustomNormalTransformFunctor {
  // This function takes in a random value and returns the corresponding sample from the normal
  // distribution
  __device__ T operator()(ComputeType random_val) const {
    T output = static_cast<T>(random_val * std_val + mean_val);
    output = mean_tensor + output * std_tensor;
    return output;
  }
  // mean_val and std_val are the mean and standard deviation of the normal distribution
  // mean_tensor and std_tensor are the mean and standard deviation tensors respectively
  ComputeType mean_val;
  ComputeType std_val;
  T mean_tensor;
  T std_tensor;

  CustomNormalTransformFunctor(ComputeType mean_val, ComputeType std_val, T mean_tensor,
                               T std_tensor) {
    this->mean_val = mean_val;
    this->std_val = std_val;
    this->mean_tensor = mean_tensor;
    this->std_tensor = std_tensor;
  }
};

}  // namespace
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_DISTRIBUTIONS_NORMAL_KERNEL_H_
        // ONEFLOW_USER_KERNELS_normal_with_tensor_kernel_h_