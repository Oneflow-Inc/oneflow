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
#ifndef ONEFLOW_USER_KERNELS_ENTR_H
#define ONEFLOW_USER_KERNELS_ENTR_H
#include <cmath>
#include <limits>
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"
namespace oneflow {
template<DeviceType device_type, typename T>
struct EntrFunctor {
  OF_DEVICE_FUNC T operator()(const T x) const;
};

template<DeviceType device_type, typename T>
struct EntrGradFunctor {
  OF_DEVICE_FUNC T operator()(const T x, const T dy) const;
};

#define REGISTER_ENTR_KERNEL(device, kernel_name, functor, out_dtype, input_a_dtype,               \
                             create_function, out_name, input_a_name)                              \
  REGISTER_USER_KERNEL(kernel_name)                                                                \
      .SetCreateFn([]() {                                                                          \
        return user_op::NewOpKernel<                                                               \
            UnaryElemwiseXpuKernel<device, functor<device, out_dtype>, out_dtype, input_a_dtype>>( \
            create_function, out_name, input_a_name);                                              \
      })                                                                                           \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceType() == device)                                                     \
          && (user_op::HobDataType(input_a_name, 0) == GetDataType<out_dtype>::value));

#define REGISTER_ENTR_GRAD_KERNEL(device, kernel_name, functor, out_dtype, input_a_dtype,  \
                                  input_b_dtype, create_function, out_name, input_a_name,  \
                                  input_b_name)                                            \
  REGISTER_USER_KERNEL(kernel_name)                                                        \
      .SetCreateFn([]() {                                                                  \
        return user_op::NewOpKernel<BinaryElemwiseXpuKernel<                               \
            device, functor<device, out_dtype>, out_dtype, input_a_dtype, input_b_dtype>>( \
            create_function, out_name, input_a_name, input_b_name);                        \
      })                                                                                   \
      .SetIsMatchedHob(                                                                    \
          (user_op::HobDeviceType() == device)                                             \
          && (user_op::HobDataType(input_a_name, 0) == GetDataType<out_dtype>::value));

#define REGISTER_ENTR_KERNEL_DEVICE_TYPE(device, type)                                             \
  REGISTER_ENTR_KERNEL(                                                                            \
      device, "entr", EntrFunctor, type, type,                                                     \
      ([](user_op::KernelComputeContext* ctx) { return EntrFunctor<device, type>(); }), "y", "x"); \
  REGISTER_ENTR_GRAD_KERNEL(                                                                       \
      device, "entr_grad", EntrGradFunctor, type, type, type,                                      \
      ([](user_op::KernelComputeContext* ctx) { return EntrGradFunctor<device, type>(); }), "dx",  \
      "x", "dy");

}  // namespace oneflow
#endif  // ONEFLOW_USER_KERNELS_ENTR_H
