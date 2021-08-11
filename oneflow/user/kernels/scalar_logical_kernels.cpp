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
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const T scalar, const T* in,
                  int8_t* out) {
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
    int8_t* out_ptr = out->mut_dptr<int8_t>();

    int64_t elem_cnt = out->shape().elem_cnt();
    if (elem_cnt != 0) {
      ScalarLogicalFunctor<device_type, BIN_OP, T>()(ctx->device_ctx(), elem_cnt, scalar_operand,
                                                     in_ptr, out_ptr);
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, kernel_name, binary_op, \
                                                           input_dtype)                    \
  REGISTER_USER_KERNEL(kernel_name)                                                        \
      .SetCreateFn<ScalarLogicalKernel<device, binary_op, input_dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                 \
                       & (user_op::HobDataType("in", 0) == GetDataType<input_dtype>::value));

#define REGISTER_SCALAR_LOGICAL_KERNEL(device, dtype)                                              \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_equal", BinaryFuncEQ, \
                                                     dtype);                                       \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_not_equal",           \
                                                     BinaryFuncNE, dtype);                         \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_greater",             \
                                                     BinaryFuncGT, dtype);                         \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_greater_equal",       \
                                                     BinaryFuncGE, dtype);                         \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_less", BinaryFuncLT,  \
                                                     dtype);                                       \
  REGISTER_UNARY_LOGICAL_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_logical_less_equal",          \
                                                     BinaryFuncLE, dtype);

#define REGISTER_SCALAR_LOGICAL_CPU_KERNELS()               \
  REGISTER_SCALAR_LOGICAL_KERNEL(DeviceType::kCPU, int32_t) \
  REGISTER_SCALAR_LOGICAL_KERNEL(DeviceType::kCPU, int64_t) \
  REGISTER_SCALAR_LOGICAL_KERNEL(DeviceType::kCPU, float)   \
  REGISTER_SCALAR_LOGICAL_KERNEL(DeviceType::kCPU, double)

#define REGISTER_SCALAR_LOGICAL_GPU_KERNELS()               \
  REGISTER_SCALAR_LOGICAL_KERNEL(DeviceType::kGPU, int32_t) \
  REGISTER_SCALAR_LOGICAL_KERNEL(DeviceType::kGPU, int64_t) \
  REGISTER_SCALAR_LOGICAL_KERNEL(DeviceType::kGPU, float)   \
  REGISTER_SCALAR_LOGICAL_KERNEL(DeviceType::kGPU, double)

REGISTER_SCALAR_LOGICAL_CPU_KERNELS();

#ifdef WITH_CUDA
REGISTER_SCALAR_LOGICAL_GPU_KERNELS();
#endif  // WITH_CUDA

}  // namespace oneflow
