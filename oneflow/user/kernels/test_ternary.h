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
#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_TERNARY_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_TERNARY_KERNEL_H_
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {

template<typename T>
struct TestTernaryFunctor {
  OF_DEVICE_FUNC T operator()(T x1, T x2, T x3) const { return (x1 + x2) * x3; }
};

#define REGISTER_TEST_TERNARY_KERNEL(device, dtype)                                                \
  REGISTER_USER_KERNEL("test_ternary")                                                             \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                         \
        return new TernaryElemwiseXpuKernel<device, TestTernaryFunctor<dtype>, dtype>(             \
            [](user_op::KernelComputeContext* ctx) { return TestTernaryFunctor<dtype>(); }, "out", \
            "in1", "in2", "in3");                                                                  \
      })                                                                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                         \
                       & (user_op::HobDataType("in1", 0) == GetDataType<dtype>::value));

}  // namespace oneflow
#endif  //
