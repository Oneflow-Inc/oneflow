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

#define REGISTER_ACTIVATION_CPU_KERNEL(dtype)                    \
  REGISTER_ELU_BACKWARD_KERNEL(DeviceType::kCPU, dtype);         \
  REGISTER_CELU_BACKWARD_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_HARDSWISH_BACKWARD_KERNEL(DeviceType::kCPU, dtype);   \
  REGISTER_HARDSIGMOID_BACKWARD_KERNEL(DeviceType::kCPU, dtype); \
  REGISTER_HARDSHRINK_BACKWARD_KERNEL(DeviceType::kCPU, dtype);  \
  REGISTER_HARDTANH_BACKWARD_KERNEL(DeviceType::kCPU, dtype);    \
  REGISTER_MISH_BACKWARD_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_RELU_BACKWARD_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_SILU_BACKWARD_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_SELU_BACKWARD_KERNEL(DeviceType::kCPU, dtype);        \
  REGISTER_SOFTSHRINK_BACKWARD_KERNEL(DeviceType::kCPU, dtype);  \
  REGISTER_SOFTSIGN_BACKWARD_KERNEL(DeviceType::kCPU, dtype);    \
  REGISTER_SOFTPLUS_BACKWARD_KERNEL(DeviceType::kCPU, dtype);    \
  REGISTER_LEAKYRELU_BACKWARD_KERNEL(DeviceType::kCPU, dtype);   \
  REGISTER_THRESHOLD_BACKWARD_KERNEL(DeviceType::kCPU, dtype);

REGISTER_ACTIVATION_CPU_KERNEL(float);
REGISTER_ACTIVATION_CPU_KERNEL(double);

REGISTER_ELU_FORWARD_KERNEL();
REGISTER_CELU_FORWARD_KERNEL();
REGISTER_HARDSWISH_FORWARD_KERNEL();
REGISTER_HARDSIGMOID_FORWARD_KERNEL();
REGISTER_HARDSHRINK_FORWARD_KERNEL();
REGISTER_HARDTANH_FORWARD_KERNEL();
REGISTER_GELU_FORWARD_KERNEL();
REGISTER_LEAKYRELU_FORWARD_KERNEL();
REGISTER_MISH_FORWARD_KERNEL();
REGISTER_RELU_FORWARD_KERNEL();
REGISTER_SILU_FORWARD_KERNEL();
REGISTER_SELU_FORWARD_KERNEL();
REGISTER_SOFTSHRINK_FORWARD_KERNEL();
REGISTER_SOFTSIGN_FORWARD_KERNEL();
REGISTER_SOFTPLUS_FORWARD_KERNEL();
REGISTER_TANH_FORWARD_KERNEL();
REGISTER_THRESHOLD_FORWARD_KERNEL();

}  // namespace oneflow
