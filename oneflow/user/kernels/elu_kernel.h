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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_ELU_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_ELU_KERNEL_H_
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {

template<typename T>
struct EluFunctor {
  OF_DEVICE_FUNC explicit EluFunctor(T alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x > static_cast<T>(0)) ? x : alpha * (exp(x) - static_cast<T>(1));
  }
  const T alpha;
};

#define REGISTER_ELU_KERNEL(device, dtype)                                              \
  REGISTER_USER_KERNEL("elu")                                                           \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                              \
        return new UnaryElemwiseXpuKernel<device, EluFunctor<dtype>, dtype>(            \
            "in", "out", [](user_op::KernelComputeContext* ctx) {                       \
              return EluFunctor<dtype>(static_cast<dtype>(ctx->Attr<double>("alpha"))); \
            });                                                                         \
      })                                                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));
}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_ELU_KERNEL_H_
