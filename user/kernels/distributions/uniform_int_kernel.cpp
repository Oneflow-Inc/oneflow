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
#include "oneflow/user/kernels/distributions/uniform_int_kernel.h"

namespace oneflow {

namespace {
#define REGISTER_UNIFORM_KERNEL(device, dtype)              \
  REGISTER_USER_KERNEL("uniform_int")                       \
      .SetCreateFn<UniformIntKernel<device, dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

REGISTER_UNIFORM_KERNEL(DeviceType::kCPU, float)
REGISTER_UNIFORM_KERNEL(DeviceType::kCPU, double)
REGISTER_UNIFORM_KERNEL(DeviceType::kCPU, uint8_t)
REGISTER_UNIFORM_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_UNIFORM_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_UNIFORM_KERNEL(DeviceType::kCPU, int64_t)
#ifdef WITH_CUDA
REGISTER_UNIFORM_KERNEL(DeviceType::kCUDA, float)
REGISTER_UNIFORM_KERNEL(DeviceType::kCUDA, double)
REGISTER_UNIFORM_KERNEL(DeviceType::kCUDA, uint8_t)
REGISTER_UNIFORM_KERNEL(DeviceType::kCUDA, int8_t)
REGISTER_UNIFORM_KERNEL(DeviceType::kCUDA, int32_t)
REGISTER_UNIFORM_KERNEL(DeviceType::kCUDA, int64_t)
#endif  // WITH_CUDA

}  // namespace

}  // namespace oneflow
