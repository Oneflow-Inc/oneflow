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
#include "oneflow/user/kernels/scalar_math_kernels.h"

namespace oneflow {

template<template<typename> class BIN_OP, typename T>
struct ScalarMathFunctor<DeviceType::kCPU, BIN_OP, T> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T scalar, const T* in, T* out) {
    DoScalarMath<BIN_OP, T>(elem_cnt, scalar, in, out);
  }
};

template<template<typename> class BIN_OP, typename T>
struct ScalarReverseMathFunctor<DeviceType::kCPU, BIN_OP, T> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T scalar, const T* in, T* out) {
    DoScalarReverseMath<BIN_OP, T>(elem_cnt, scalar, in, out);
  }
};

template<DeviceType device_type, template<typename> class BIN_OP, typename T>
class ScalarMathKernel final : public user_op::OpKernel {
 public:
  ScalarMathKernel() = default;
  ~ScalarMathKernel() = default;

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
    T* out_ptr = out->mut_dptr<T>();

    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      ScalarMathFunctor<device_type, BIN_OP, T>()(ctx->stream(), elem_cnt, scalar_operand, in_ptr,
                                                  out_ptr);
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, template<typename> class BIN_OP, typename T>
class ScalarReverseMathKernel final : public user_op::OpKernel {
 public:
  ScalarReverseMathKernel() = default;
  ~ScalarReverseMathKernel() = default;

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
    T* out_ptr = out->mut_dptr<T>();

    int64_t elem_cnt = out->shape_view().elem_cnt();
    if (elem_cnt != 0) {
      ScalarReverseMathFunctor<device_type, BIN_OP, T>()(ctx->stream(), elem_cnt, scalar_operand,
                                                         in_ptr, out_ptr);
    } else {
      // For 0-d Tensor
      return;
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(device, kernel_name, binary_op,       \
                                                        input_dtype_pair)                     \
  REGISTER_USER_KERNEL(kernel_name)                                                           \
      .SetCreateFn<ScalarMathKernel<device, binary_op, OF_PP_PAIR_FIRST(input_dtype_pair)>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                   \
                       && (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(input_dtype_pair)));

#define REGISTER_SCALAR_MATH_KERNEL(device, dtype_pair)                                          \
  REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_add", BinaryFuncAdd,           \
                                                  dtype_pair);                                   \
  REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_floordiv", BinaryFuncFloorDiv, \
                                                  dtype_pair);                                   \
  REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_fmod", BinaryFuncFMod,         \
                                                  dtype_pair);                                   \
  REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_mul", BinaryFuncMul,           \
                                                  dtype_pair);                                   \
  REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_div", BinaryFuncDiv,           \
                                                  dtype_pair);                                   \
  REGISTER_UNARY_MATH_SCALAR_ELEMWISE_USER_KERNEL(device, "scalar_pow", BinaryFuncPow, dtype_pair);

#define REGISTER_UNARY_MATH_SCALAR_REVERSE_ELEMWISE_USER_KERNEL(device, kernel_name, binary_op, \
                                                                input_dtype_pair)               \
  REGISTER_USER_KERNEL(kernel_name)                                                             \
      .SetCreateFn<                                                                             \
          ScalarReverseMathKernel<device, binary_op, OF_PP_PAIR_FIRST(input_dtype_pair)>>()     \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                     \
                       && (user_op::HobDataType("in", 0) == OF_PP_PAIR_SECOND(input_dtype_pair)));

#define REGISTER_SCALAR_REVERSE_POW_KERNEL(device, dtype_pair)                          \
  REGISTER_UNARY_MATH_SCALAR_REVERSE_ELEMWISE_USER_KERNEL(device, "scalar_reverse_pow", \
                                                          BinaryFuncPow, dtype_pair);

// we register uint8_t, int8_t, int32_t, int64_t, float, double, float16.
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_MATH_KERNEL, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     FLOAT16_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_REVERSE_POW_KERNEL, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     FLOAT16_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_MATH_KERNEL, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     FLOAT16_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCALAR_REVERSE_POW_KERNEL, (DeviceType::kCUDA),
                                 ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ
                                     FLOAT16_DATA_TYPE_SEQ)
#endif  // WITH_CUDA

template<DeviceType device_type, typename T>
class CpuScalarPowGradKernel final : public user_op::OpKernel {
 public:
  CpuScalarPowGradKernel() = default;
  ~CpuScalarPowGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* x_ptr = x_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }

    const int32_t elem_cnt = x_tensor->shape_view().elem_cnt();
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] =
          scalar_operand * (std::pow(x_ptr[i], scalar_operand - static_cast<T>(1))) * dy_ptr[i];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(device, dtype)  \
  REGISTER_USER_KERNEL("scalar_pow_grad")                   \
      .SetCreateFn<CpuScalarPowGradKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(DeviceType::kCPU, double);

template<DeviceType device_type, typename T>
class CpuScalarReversePowGradKernel final : public user_op::OpKernel {
 public:
  CpuScalarReversePowGradKernel() = default;
  ~CpuScalarReversePowGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* x_ptr = x_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }

    const int32_t elem_cnt = x_tensor->shape_view().elem_cnt();
    // NOTE: y = a^x    ==>>   dy/dx = a^x * lna
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] = std::pow(scalar_operand, x_ptr[i]) * std::log(scalar_operand) * dy_ptr[i];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SCALAR_REVERSE_POW_GRAD_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("scalar_reverse_pow_grad")                  \
      .SetCreateFn<CpuScalarReversePowGradKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)        \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SCALAR_REVERSE_POW_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_CPU_SCALAR_REVERSE_POW_GRAD_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
