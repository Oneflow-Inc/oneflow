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
#ifndef _ONEFLOW_USER_KERNELS_SCALAR_MATH_KERNELS_H_
#define _ONEFLOW_USER_KERNELS_SCALAR_MATH_KERNELS_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

#define INSTANTIATE_SCALAR_MATH_FUNCTORS(device_type, binary_op)      \
  template struct ScalarMathFunctor<device_type, binary_op, uint8_t>; \
  template struct ScalarMathFunctor<device_type, binary_op, int8_t>;  \
  template struct ScalarMathFunctor<device_type, binary_op, int32_t>; \
  template struct ScalarMathFunctor<device_type, binary_op, int64_t>; \
  template struct ScalarMathFunctor<device_type, binary_op, float>;   \
  template struct ScalarMathFunctor<device_type, binary_op, double>;  \
  template struct ScalarMathFunctor<device_type, binary_op, float16>;

#define INSTANTIATE_SCALAR_REVERSE_MATH_FUNCTORS(device_type, binary_op)     \
  template struct ScalarReverseMathFunctor<device_type, binary_op, uint8_t>; \
  template struct ScalarReverseMathFunctor<device_type, binary_op, int8_t>;  \
  template struct ScalarReverseMathFunctor<device_type, binary_op, int32_t>; \
  template struct ScalarReverseMathFunctor<device_type, binary_op, int64_t>; \
  template struct ScalarReverseMathFunctor<device_type, binary_op, float>;   \
  template struct ScalarReverseMathFunctor<device_type, binary_op, double>;  \
  template struct ScalarReverseMathFunctor<device_type, binary_op, float16>;

template<DeviceType device_type, template<typename> class BIN_OP, typename T>
struct ScalarMathFunctor final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T scalar, const T* in, T* out);
};

template<template<typename> class UnaryFunctor, typename T>
OF_DEVICE_FUNC void DoScalarMath(const int64_t elem_cnt, const T scalar, const T* in, T* out) {
  XPU_1D_KERNEL_LOOP(idx, elem_cnt) { out[idx] = UnaryFunctor<T>::Invoke(in[idx], scalar); }
}

template<DeviceType device_type, template<typename> class BIN_OP, typename T>
struct ScalarReverseMathFunctor final {
  void operator()(ep::Stream* stream, const int64_t elem_cnt, const T scalar, const T* in, T* out);
};

template<template<typename> class UnaryFunctor, typename T>
OF_DEVICE_FUNC void DoScalarReverseMath(const int64_t elem_cnt, const T scalar, const T* in,
                                        T* out) {
  XPU_1D_KERNEL_LOOP(idx, elem_cnt) { out[idx] = UnaryFunctor<T>::Invoke(scalar, in[idx]); }
}

}  // namespace oneflow

#endif  // _ONEFLOW_USER_KERNELS_SCALAR_MATH_KERNELS_H_
