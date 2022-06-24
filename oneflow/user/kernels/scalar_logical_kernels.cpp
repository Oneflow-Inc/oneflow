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

namespace oneflow {

template<template<typename T> class BIN_OP, typename T>
struct ScalarLogicalFunctor<DeviceType::kCPU, BIN_OP, T> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T scalar, const T* in,
                  bool* out) {
    DoScalarLogical<BIN_OP, T>(elem_cnt, scalar, in, out);
  }
};

template<DeviceType device_type, template<typename> class BIN_OP, typename T>
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
    bool* out_ptr = out->mut_dptr<bool>();

    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      ScalarLogicalFunctor<device_type, BIN_OP, T>()(ctx->stream(), elem_cnt, scalar_operand,
                                                     in_ptr, out_ptr);
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, kernel_name, binary_op,       \
                                                           input_dtype_pair)                     \
  REGISTER_USER_KERNEL(kernel_name)                                                              \
      .SetCreateFn<ScalarLogicalKernel<device, binary_op, OF_PP_PAIR_FIRST(input_dtype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                      \
                       && (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(input_dtype_pair)));

#define REGISTER_SCALAR_LOGICAL_KERNEL(device, dtype_pair)                                         \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_equal", BinaryFuncEQ, \
                                                     dtype_pair);                                  \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_not_equal",           \
                                                     BinaryFuncNE, dtype_pair);                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_greater",             \
                                                     BinaryFuncGT, dtype_pair);                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_greater_equal",       \
                                                     BinaryFuncGE, dtype_pair);                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_less", BinaryFuncLT,  \
                                                     dtype_pair);                                  \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_less_equal",          \
                                                     BinaryFuncLE, dtype_pair);                    \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_or", BinaryFuncOR,    \
                                                     dtype_pair);                                  \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_xor", BinaryFuncXOR,  \
                                                     dtype_pair);                                  \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_and", BinaryFuncAND,  \
                                                     dtype_pair);

// we register bool, uint8_t, int8_t, int32_t, int64_t, float, double.
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_LOGICAL_KERNEL, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     BOOL_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_LOGICAL_KERNEL, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     BOOL_DATA_TYPE_SEQ)
#endif  // WITH_CUDA

}  // namespace oneflow
