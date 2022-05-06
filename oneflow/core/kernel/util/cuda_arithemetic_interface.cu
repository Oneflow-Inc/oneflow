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
#include "oneflow/core/kernel/util/cuda_arithemetic_interface.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/kernel/util/host_arithemetic_interface.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
struct AddOp {
  __device__ T operator()(T a, T b) const { return a + b; }
};

template<typename T>
struct SubOp {
  __device__ T operator()(T a, T b) const { return a - b; }
};

template<typename T>
struct MulOp {
  __device__ T operator()(T a, T b) const { return a * b; }
};

template<typename T>
struct DivOp {
  __device__ T operator()(T a, T b) const { return a / b; }
};

template<template<typename> class Op, typename T>
struct UnaryByScalarFunctor {
  __host__ __device__ explicit UnaryByScalarFunctor(T scalar) : scalar(scalar) {}
  __device__ T operator()(T a) const { return Op<T>()(a, scalar); }
  const T scalar;
};

template<template<typename> class Op, typename T>
struct UnaryByScalarPtrFunctorFactory {
  __host__ __device__ explicit UnaryByScalarPtrFunctorFactory(const T* scalar_ptr)
      : scalar_ptr(scalar_ptr) {}
  __device__ UnaryByScalarFunctor<Op, T> operator()() const {
    return UnaryByScalarFunctor<Op, T>(*scalar_ptr);
  }
  const T* scalar_ptr;
};

template<template<typename> class Op, typename T>
void LaunchUnaryByScalar(ep::Stream* stream, const int64_t n, const T* x, const T y, T* z) {
  OF_CUDA_CHECK((cuda::elementwise::Unary(UnaryByScalarFunctor<Op, T>(y), n, z, x,
                                          stream->As<ep::CudaStream>()->cuda_stream())));
}

template<template<typename> class Op, typename T>
void LaunchUnaryByScalarPtr(ep::Stream* stream, const int64_t n, const T* x, const T* y, T* z) {
  OF_CUDA_CHECK(
      (cuda::elementwise::UnaryWithFactory(UnaryByScalarPtrFunctorFactory<Op, T>(y), n, z, x,
                                           stream->As<ep::CudaStream>()->cuda_stream())));
}

}  // namespace

#define OP_BY_SCALAR(op, T)                                                                \
  void ArithemeticIf<DeviceType::kCUDA>::op##ByScalar(ep::Stream* stream, const int64_t n, \
                                                      const T* x, const T y, T* z) {       \
    LaunchUnaryByScalar<op##Op, T>(stream, n, x, y, z);                                    \
  }

#define OP_BY_SCALAR_HALF(op)                                                               \
  void ArithemeticIf<DeviceType::kCUDA>::op##ByScalar(                                      \
      ep::Stream* stream, const int64_t n, const float16* x, const float16 y, float16* z) { \
    LaunchUnaryByScalar<op##Op, half>(stream, n, reinterpret_cast<const half*>(x),          \
                                      float16_2half(y), reinterpret_cast<half*>(z));        \
  }

#define DEFINE_OP_BY_SCALAR(op) \
  OP_BY_SCALAR(op, float)       \
  OP_BY_SCALAR(op, double)      \
  OP_BY_SCALAR(op, int8_t)      \
  OP_BY_SCALAR(op, int32_t)     \
  OP_BY_SCALAR(op, int64_t)     \
  OP_BY_SCALAR_HALF(op)

DEFINE_OP_BY_SCALAR(Mul)
DEFINE_OP_BY_SCALAR(Add)

#define OP_BY_SCALAR_PTR(op, T)                                                               \
  void ArithemeticIf<DeviceType::kCUDA>::op##ByScalarPtr(ep::Stream* stream, const int64_t n, \
                                                         const T* x, const T* y, T* z) {      \
    LaunchUnaryByScalarPtr<op##Op, T>(stream, n, x, y, z);                                    \
  }

#define OP_BY_SCALAR_PTR_HALF(op)                                                            \
  void ArithemeticIf<DeviceType::kCUDA>::op##ByScalarPtr(                                    \
      ep::Stream* stream, const int64_t n, const float16* x, const float16* y, float16* z) { \
    LaunchUnaryByScalarPtr<op##Op, half>(stream, n, reinterpret_cast<const half*>(x),        \
                                         reinterpret_cast<const half*>(y),                   \
                                         reinterpret_cast<half*>(z));                        \
  }

#define DEFINE_OP_BY_SCALAR_PTR(op) \
  OP_BY_SCALAR_PTR(op, float)       \
  OP_BY_SCALAR_PTR(op, double)      \
  OP_BY_SCALAR_PTR(op, int8_t)      \
  OP_BY_SCALAR_PTR(op, int32_t)     \
  OP_BY_SCALAR_PTR(op, int64_t)     \
  OP_BY_SCALAR_PTR_HALF(op)

DEFINE_OP_BY_SCALAR_PTR(Mul)
DEFINE_OP_BY_SCALAR_PTR(Add)
DEFINE_OP_BY_SCALAR_PTR(Sub)
DEFINE_OP_BY_SCALAR_PTR(Div)

}  // namespace oneflow
