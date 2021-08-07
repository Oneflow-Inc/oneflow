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
#ifndef _ONEFLOW_USER_KERNELS_SCALAR_LOGICAL_KERNELS_H_
#define _ONEFLOW_USER_KERNELS_SCALAR_LOGICAL_KERNELS_H_
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T>
struct ScalarLogicalEqualFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalEqualFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const { return BinaryFuncEQ<T>::Invoke(x, scalar); }
  const T scalar;
};

template<typename T>
struct ScalarLogicalNotEqualFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalNotEqualFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const { return BinaryFuncNE<T>::Invoke(x, scalar); }
  const T scalar;
};

template<typename T>
struct ScalarLogicalGreaterFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalGreaterFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const { return BinaryFuncGT<T>::Invoke(x, scalar); }
  const T scalar;
};

template<typename T>
struct ScalarLogicalGreaterEqualFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalGreaterEqualFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const { return BinaryFuncGE<T>::Invoke(x, scalar); }
  const T scalar;
};

template<typename T>
struct ScalarLogicalLessFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalLessFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const { return BinaryFuncLT<T>::Invoke(x, scalar); }
  const T scalar;
};

template<typename T>
struct ScalarLogicalLessEqualFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalLessEqualFunctor(double scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const { return BinaryFuncLE<T>::Invoke(x, scalar); }
  const T scalar;
};

#define REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                                      \
    device, kernel_name, functor, out_dtype, input_a_dtype, create_function, out_name,           \
    input_a_name)                                                                                \
  REGISTER_USER_KERNEL(kernel_name)                                                              \
      .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                       \
        return new UnaryElemwiseXpuKernel<device, functor<out_dtype>, out_dtype, input_a_dtype>( \
            create_function, out_name, input_a_name);                                            \
      })                                                                                         \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceTag() == device)                                                    \
          & (user_op::HobDataType(input_a_name, 0) == GetDataType<input_a_dtype>::value));

#define REGISTER_SCALAR_LOGICAL_EQUAL_KERNEL(device, dtype)                     \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                           \
      device, "scalar_logical_equal", ScalarLogicalEqualFunctor, int8_t, dtype, \
      [](user_op::KernelComputeContext* ctx) {                                  \
        return ScalarLogicalEqualFunctor<int8_t>(ctx->Attr<double>("scalar"));  \
      },                                                                        \
      "out", "in");

#define REGISTER_SCALAR_LOGICAL_NOTEQUAL_KERNEL(device, dtype)                         \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                                  \
      device, "scalar_logical_not_equal", ScalarLogicalNotEqualFunctor, int8_t, dtype, \
      [](user_op::KernelComputeContext* ctx) {                                         \
        return ScalarLogicalNotEqualFunctor<int8_t>(ctx->Attr<double>("scalar"));      \
      },                                                                               \
      "out", "in");

#define REGISTER_SCALAR_LOGICAL_GREATER_KERNEL(device, dtype)                       \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                               \
      device, "scalar_logical_greater", ScalarLogicalGreaterFunctor, int8_t, dtype, \
      [](user_op::KernelComputeContext* ctx) {                                      \
        return ScalarLogicalGreaterFunctor<int8_t>(ctx->Attr<double>("scalar"));    \
      },                                                                            \
      "out", "in");

#define REGISTER_SCALAR_LOGICAL_GREATER_EQUAL_KERNEL(device, dtype)                            \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                                          \
      device, "scalar_logical_greater_equal", ScalarLogicalGreaterEqualFunctor, int8_t, dtype, \
      [](user_op::KernelComputeContext* ctx) {                                                 \
        return ScalarLogicalGreaterEqualFunctor<int8_t>(ctx->Attr<double>("scalar"));          \
      },                                                                                       \
      "out", "in");

#define REGISTER_SCALAR_LOGICAL_LESS_KERNEL(device, dtype)                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                         \
      device, "scalar_logical_less", ScalarLogicalLessFunctor, int8_t, dtype, \
      [](user_op::KernelComputeContext* ctx) {                                \
        return ScalarLogicalLessFunctor<int8_t>(ctx->Attr<double>("scalar")); \
      },                                                                      \
      "out", "in");

#define REGISTER_SCALAR_LOGICAL_LESS_EQUAL_KERNEL(device, dtype)                         \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                                    \
      device, "scalar_logical_less_equal", ScalarLogicalLessEqualFunctor, int8_t, dtype, \
      [](user_op::KernelComputeContext* ctx) {                                           \
        return ScalarLogicalLessEqualFunctor<int8_t>(ctx->Attr<double>("scalar"));       \
      },                                                                                 \
      "out", "in");

}  // namespace oneflow

#endif  //_ONEFLOW_USER_KERNELS_SCALAR_LOGICAL_KERNELS_H_
