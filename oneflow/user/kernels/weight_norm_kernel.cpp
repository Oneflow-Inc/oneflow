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

namespace oneflow {

template<typename T>
class WeightNormKernel final : public user_op::OpKernel {
 public:
  WeightNormKernel() = default;
  ~WeightNormKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override { TODO(); };
};

#define REGISTER_WEIGHT_NORM_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("weight_norm")                     \
      .SetCreateFn<WeightNormKernel<dtype>>()             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("v", 0) == GetDataType<dtype>::value));

REGISTER_WEIGHT_NORM_KERNEL(float)
REGISTER_WEIGHT_NORM_KERNEL(double)

}  // namespace oneflow