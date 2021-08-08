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
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<template<typename> class UnaryFunctor, typename T>
struct ScalarLogicalFunctor {
  OF_DEVICE_FUNC explicit ScalarLogicalFunctor(T scalar) : scalar(scalar) {}
  OF_DEVICE_FUNC int8_t operator()(T x) const { return UnaryFunctor<T>::Invoke(x, scalar); }
  const T scalar;
};

template<DeviceType device_type, typename T, template<typename> class BIN_OP>
class ScalarLogicalKernel final : public user_op::OpKernel {
 public:
  ScalarLogicalKernel() = default;
  ~ScalarLogicalKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    const T* in_ptr = in->dptr<T>();
    int8_t* out_ptr = out->mut_dptr<int8_t>();

    int64_t elem_cnt = out->shape().elem_cnt();

    ScalarLogicalFunctor<BIN_OP, T> functor(scalar_operand); 

    XPU_1D_KERNEL_LOOP(i, elem_cnt) { out_ptr[i] = functor(in_ptr[i]); }

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, kernel_name, binary_op, input_dtype)  \
  REGISTER_USER_KERNEL(kernel_name)                                                              \
      .SetCreateFn<ScalarLogicalKernel<device, input_dtype, binary_op>>()              \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceTag() == device)                                                    \
          & (user_op::HobDataType("in", 0) == GetDataType<input_dtype>::value));                 \

// #define REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, kernel_name, binary_op, input_dtype)  \
//   REGISTER_USER_KERNEL(kernel_name)                                                              \
//       .SetCreateFn([](user_op::KernelCreateContext* ctx) {                                       \
//         return new UnaryElemwiseXpuKernel<device, ScalarLogicalFunctor<binary_op, int8_t>, int8_t, input_dtype>( \
//             [](user_op::KernelComputeContext* ctx) {return ScalarLogicalFunctor<binary_op, int8_t>(ctx->Attr<double>("float_operand")); }, \
//             "out", "in");                                            \
//       })                                                                                         \
//       .SetIsMatchedHob(                                                                          \
//           (user_op::HobDeviceTag() == device)                                                    \
//           & (user_op::HobDataType("in", 0) == GetDataType<input_dtype>::value));                 \

#define REGISTER_SCALAR_LOGICAL_EQUAL_KERNEL(device, dtype)                     \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                           \
      device, "scalar_logical_equal", BinaryFuncEQ, dtype);

#define REGISTER_SCALAR_LOGICAL_NOTEQUAL_KERNEL(device, dtype)                     \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                           \
      device, "scalar_logical_not_equal", BinaryFuncNE, dtype);

#define REGISTER_SCALAR_LOGICAL_GREATER_KERNEL(device, dtype)                       \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                           \
      device, "scalar_logical_greater", BinaryFuncGT, dtype);

#define REGISTER_SCALAR_LOGICAL_GREATER_EQUAL_KERNEL(device, dtype)                            \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                           \
      device, "scalar_logical_greater_equal", BinaryFuncGE, dtype);

#define REGISTER_SCALAR_LOGICAL_LESS_KERNEL(device, dtype)                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                           \
      device, "scalar_logical_less", BinaryFuncLT, dtype);


#define REGISTER_SCALAR_LOGICAL_LESS_EQUAL_KERNEL(device, dtype)                         \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(                           \
      device, "scalar_logical_less_equal", BinaryFuncLE, dtype);

}  // namespace oneflow

#endif  //_ONEFLOW_USER_KERNELS_SCALAR_LOGICAL_KERNELS_H_
