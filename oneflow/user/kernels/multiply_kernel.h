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
#ifndef ONEFLOW_USER_KERNELS_MULTIPLY_KERNEL_H_
#define ONEFLOW_USER_KERNELS_MULTIPLY_KERNEL_H_

#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {

template<typename T>
struct MultiplyFunctor {
  OF_DEVICE_FUNC T operator()(T a, T b) const { return a * b; }
};

#define REGISTER_MULTIPLY_KERNEL(device, dtype)                                                \
  REGISTER_BINARY_ELEMWISE_USER_KERNEL(                                                        \
      device, "multiply", MultiplyFunctor, dtype, dtype, dtype,                                \
      [](user_op::KernelComputeContext* ctx) { return MultiplyFunctor<dtype>(); }, "out", "x", \
      "y");

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_MULTIPLY_KERNEL_H_
