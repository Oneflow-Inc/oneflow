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
#include "oneflow/user/kernels/scalar_logical_kernels.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"

namespace oneflow {

#define REGISTER_SCALAR_LOGICAL_GPU_KERNEL(dtype)                        \
  REGISTER_SCALAR_LOGICAL_EQUAL_KERNEL(DeviceType::kGPU, dtype);         \
  REGISTER_SCALAR_LOGICAL_NOTEQUAL_KERNEL(DeviceType::kGPU, dtype);      \
  REGISTER_SCALAR_LOGICAL_GREATER_KERNEL(DeviceType::kGPU, dtype);       \
  REGISTER_SCALAR_LOGICAL_GREATER_EQUAL_KERNEL(DeviceType::kGPU, dtype); \
  REGISTER_SCALAR_LOGICAL_LESS_KERNEL(DeviceType::kGPU, dtype);          \
  REGISTER_SCALAR_LOGICAL_LESS_EQUAL_KERNEL(DeviceType::kGPU, dtype);

REGISTER_SCALAR_LOGICAL_GPU_KERNEL(float);
REGISTER_SCALAR_LOGICAL_GPU_KERNEL(double);

}  // namespace oneflow