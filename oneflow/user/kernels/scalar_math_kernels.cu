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
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

template<template<typename> class Op, typename T>
struct UnaryByScalarFunctor {
  __host__ __device__ explicit UnaryByScalarFunctor(T scalar) : scalar(scalar) {}
  __device__ T operator()(T a) const { return Op<T>::Invoke(a, scalar); }
  const T scalar;
};

template<template<typename> class Op, typename T>
struct UnaryByScalarReverseFunctor {
  __host__ __device__ explicit UnaryByScalarReverseFunctor(T scalar) : scalar(scalar) {}
  __device__ T operator()(T a) const { return Op<T>::Invoke(scalar, a); }
  const T scalar;
};

template<template<typename> class Op>
struct UnaryByScalarFunctor<Op, float16> {
  __host__ __device__ explicit UnaryByScalarFunctor(half scalar) : scalar(scalar) {}
  __device__ half operator()(half a) const { return Op<half>::Invoke(a, scalar); }
  const half scalar;
};

template<template<typename> class Op>
struct UnaryByScalarReverseFunctor<Op, float16> {
  __host__ __device__ explicit UnaryByScalarReverseFunctor(half scalar) : scalar(scalar) {}
  __device__ half operator()(half a) const { return Op<half>::Invoke(scalar, a); }
  const half scalar;
};

template<template<typename> class BIN_OP, typename T>
struct ScalarMathFunctor<DeviceType::kCUDA, BIN_OP, T> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T scalar, const T* in, T* out) {
    OF_CUDA_CHECK(cuda::elementwise::Unary(UnaryByScalarFunctor<BIN_OP, T>(scalar), elem_cnt, out,
                                           in, stream->As<ep::CudaStream>()->cuda_stream()));
  }
};

template<template<typename> class BIN_OP>
struct ScalarMathFunctor<DeviceType::kCUDA, BIN_OP, float16> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, float16 scalar, const float16* in,
                  float16* out) {
    OF_CUDA_CHECK(cuda::elementwise::Unary(
        UnaryByScalarFunctor<BIN_OP, float16>(float16_2half(scalar)), elem_cnt,
        reinterpret_cast<half*>(out), reinterpret_cast<const half*>(in),
        stream->As<ep::CudaStream>()->cuda_stream()));
  }
};

template<template<typename> class BIN_OP, typename T>
struct ScalarReverseMathFunctor<DeviceType::kCUDA, BIN_OP, T> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T scalar, const T* in, T* out) {
    OF_CUDA_CHECK(cuda::elementwise::Unary(UnaryByScalarReverseFunctor<BIN_OP, T>(scalar), elem_cnt,
                                           out, in, stream->As<ep::CudaStream>()->cuda_stream()));
  }
};

template<template<typename> class BIN_OP>
struct ScalarReverseMathFunctor<DeviceType::kCUDA, BIN_OP, float16> final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, float16 scalar, const float16* in,
                  float16* out) {
    OF_CUDA_CHECK(cuda::elementwise::Unary(
        UnaryByScalarReverseFunctor<BIN_OP, float16>(float16_2half(scalar)), elem_cnt,
        reinterpret_cast<half*>(out), reinterpret_cast<const half*>(in),
        stream->As<ep::CudaStream>()->cuda_stream()));
  }
};

INSTANTIATE_SCALAR_MATH_FUNCTORS(DeviceType::kCUDA, BinaryFuncAdd);
INSTANTIATE_SCALAR_MATH_FUNCTORS(DeviceType::kCUDA, BinaryFuncFloorDiv);
INSTANTIATE_SCALAR_MATH_FUNCTORS(DeviceType::kCUDA, BinaryFuncFMod);
INSTANTIATE_SCALAR_MATH_FUNCTORS(DeviceType::kCUDA, BinaryFuncMul);
INSTANTIATE_SCALAR_MATH_FUNCTORS(DeviceType::kCUDA, BinaryFuncDiv);
INSTANTIATE_SCALAR_MATH_FUNCTORS(DeviceType::kCUDA, BinaryFuncPow);
INSTANTIATE_SCALAR_REVERSE_MATH_FUNCTORS(DeviceType::kCUDA, BinaryFuncPow);

template<typename T>
struct ScalarPowGradFunctor {
  OF_DEVICE_FUNC explicit ScalarPowGradFunctor(T exponent) : exponent(exponent) {}
  __device__ T operator()(T x, T dy) const {
    return exponent * (pow(x, exponent - static_cast<T>(1.0))) * dy;
  }
  const T exponent;
};

template<>
struct ScalarPowGradFunctor<half> {
  OF_DEVICE_FUNC explicit ScalarPowGradFunctor(half exponent) : exponent(exponent) {}
  __device__ half operator()(half x, half dy) const {
    return __float2half(__half2float(exponent)
                        * (powf(__half2float(x), __half2float(exponent) - static_cast<float>(1.0)))
                        * __half2float(dy));
  }
  const half exponent;
};

template<typename T>
struct ScalarReversePowGradFunctor {
  OF_DEVICE_FUNC explicit ScalarReversePowGradFunctor(T exponent) : exponent(exponent) {}
  __device__ T operator()(T x, T dy) const { return pow(exponent, x) * log(exponent) * dy; }
  const T exponent;
};

template<>
struct ScalarReversePowGradFunctor<float> {
  OF_DEVICE_FUNC explicit ScalarReversePowGradFunctor(float exponent) : exponent(exponent) {}
  __device__ float operator()(float x, float dy) const {
    return powf(exponent, x) * logf(exponent) * dy;
  }
  const float exponent;
};

template<>
struct ScalarReversePowGradFunctor<half> {
  OF_DEVICE_FUNC explicit ScalarReversePowGradFunctor(half exponent) : exponent(exponent) {}
  __device__ half operator()(half x, half dy) const {
    const float exp = __half2float(exponent);
    return __float2half(exp * powf(exp, __half2float(x)) * logf(exp) * __half2float(dy));
  }
  const half exponent;
};

template<DeviceType device_type, typename T>
class GpuScalarPowGradKernel final : public user_op::OpKernel {
 public:
  GpuScalarPowGradKernel() = default;
  ~GpuScalarPowGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
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
    OF_CUDA_CHECK((oneflow::cuda::elementwise::Binary(
        ScalarPowGradFunctor<T>(scalar_operand), elem_cnt, dx_ptr, x_ptr, dy_ptr,
        ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_SCALAR_POW_BACKWARD_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("scalar_pow_grad")                       \
      .SetCreateFn<GpuScalarPowGradKernel<device, dtype>>()     \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_SCALAR_POW_BACKWARD_KERNEL(DeviceType::kCUDA, float);
REGISTER_CUDA_SCALAR_POW_BACKWARD_KERNEL(DeviceType::kCUDA, double);

template<DeviceType device_type, typename T>
class GpuScalarReversePowGradKernel final : public user_op::OpKernel {
 public:
  GpuScalarReversePowGradKernel() = default;
  ~GpuScalarReversePowGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
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
    OF_CUDA_CHECK((oneflow::cuda::elementwise::Binary(
        ScalarReversePowGradFunctor<T>(scalar_operand), elem_cnt, dx_ptr, x_ptr, dy_ptr,
        ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_SCALAR_REVERSE_POW_BACKWARD_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("scalar_reverse_pow_grad")                       \
      .SetCreateFn<GpuScalarReversePowGradKernel<device, dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)             \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_SCALAR_REVERSE_POW_BACKWARD_KERNEL(DeviceType::kCUDA, float);
REGISTER_CUDA_SCALAR_REVERSE_POW_BACKWARD_KERNEL(DeviceType::kCUDA, double);

}  // namespace oneflow
