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
#include "oneflow/user/kernels/elementwise_maximum_minimum_kernel_util.h"

namespace oneflow {
namespace user_op {

#define REGISTER_MAXIMUM_KERNELS(device, dtype)                                          \
  REGISTER_USER_KERNEL("elementwise_maximum")                                            \
      .SetCreateFn<ElemwiseXimumKernel<device, MaximumForwardFunctor, dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));   \
  REGISTER_USER_KERNEL("elementwise_maximum_backward")                                   \
      .SetCreateFn<ElemwiseXimumBackwardKernel<device, MaximumBackwardFunctor, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_MINIMUM_KERNELS(device, dtype)                                          \
  REGISTER_USER_KERNEL("elementwise_minimum")                                            \
      .SetCreateFn<ElemwiseXimumKernel<device, MinimumForwardFunctor, dtype>>()          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));   \
  REGISTER_USER_KERNEL("elementwise_minimum_backward")                                   \
      .SetCreateFn<ElemwiseXimumBackwardKernel<device, MinimumBackwardFunctor, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)     \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_MAXIMUM_KERNELS(DeviceType::kCPU, float);
REGISTER_MAXIMUM_KERNELS(DeviceType::kCPU, double);
REGISTER_MINIMUM_KERNELS(DeviceType::kCPU, float);
REGISTER_MINIMUM_KERNELS(DeviceType::kCPU, double);

#ifdef WITH_CUDA
REGISTER_MAXIMUM_KERNELS(DeviceType::kGPU, float);
REGISTER_MAXIMUM_KERNELS(DeviceType::kGPU, double);
REGISTER_MINIMUM_KERNELS(DeviceType::kGPU, float);
REGISTER_MINIMUM_KERNELS(DeviceType::kGPU, double);
#endif
}  // namespace user_op
}  // namespace oneflow
