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
#include "oneflow/user/kernels/activation_kernels.h"

namespace oneflow {

#define REGISTER_ACTIVATION_CPU_KERNEL(dtype)           \
  REGISTER_ELU_KERNEL(DeviceType::kCPU, dtype);         \
  REGISTER_CELU_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_HARDSWISH_KERNEL(DeviceType::kCPU, dtype);   \
  REGISTER_HARDSIGMOID_KERNEL(DeviceType::kCPU, dtype); \
  REGISTER_HARDSHRINK_KERNEL(DeviceType::kCPU, dtype);  \
  REGISTER_HARDTANH_KERNEL(DeviceType::kCPU, dtype);    \
  REGISTER_MISH_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_SILU_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_SELU_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_SOFTSHRINK_KERNEL(DeviceType::kCPU, dtype);  \
  REGISTER_SOFTSIGN_KERNEL(DeviceType::kCPU, dtype);    \
  REGISTER_SOFTPLUS_KERNEL(DeviceType::kCPU, dtype);    \
  REGISTER_LEAKYRELU_KERNEL(DeviceType::kCPU, dtype);   \
  REGISTER_THRESHOLD_KERNEL(DeviceType::kCPU, dtype);   \
  REGISTER_RELU_BACKWARD_KERNEL(DeviceType::kCPU, dtype);

REGISTER_ACTIVATION_CPU_KERNEL(float);
REGISTER_ACTIVATION_CPU_KERNEL(double);

}  // namespace oneflow
