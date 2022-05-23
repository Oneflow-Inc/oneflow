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
#ifndef ONEFLOW_USER_KERNELS_SPECIAL_H
#define ONEFLOW_USER_KERNELS_SPECIAL_H
#include <cmath>
#include <limits>
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"
namespace oneflow {

#define SPECIAL_UNARY_OPS OF_PP_MAKE_TUPLE_SEQ("entr", Entr)

#define DECL_SPECIAL_OPS_FUNCTORS(placeholder, functor_name) \
  template<DeviceType device_type, typename T>               \
  struct functor_name##Functor;                              \
  template<DeviceType device_type, typename T>               \
  struct functor_name##GradFunctor;

OF_PP_FOR_EACH_TUPLE(DECL_SPECIAL_OPS_FUNCTORS, SPECIAL_UNARY_OPS)
#undef DECL_SPECIAL_OPS_FUNCTORS

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

#define REGISTER_SPECIAL_OPS_KERNEL_DEVICE_TYPE(kernel_name, func_prefix, device, type)          \
  REGISTER_ENTR_KERNEL(                                                                          \
      device, kernel_name, func_prefix##Functor, type, type,                                     \
      ([](user_op::KernelComputeContext* ctx) { return func_prefix##Functor<device, type>(); }), \
      "y", "x");                                                                                 \
  REGISTER_ENTR_GRAD_KERNEL(device, kernel_name "_grad", func_prefix##GradFunctor, type, type,   \
                            type, ([](user_op::KernelComputeContext* ctx) {                      \
                              return func_prefix##GradFunctor<device, type>();                   \
                            }),                                                                  \
                            "dx", "x", "dy");

}  // namespace oneflow
#endif  // ONEFLOW_USER_KERNELS_SPECIAL_H
