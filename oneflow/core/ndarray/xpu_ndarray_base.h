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
#ifndef ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BASE_H_
#define ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BASE_H_

#include "oneflow/core/ndarray/xpu_shape.h"

namespace oneflow {

template<typename T, template<typename> class unary_func, typename X>
class XpuUnaryFuncNdarray;
template<typename T, template<typename> class binary_func, typename A, typename B>
class XpuBinaryFuncNdarray;
template<typename T>
class XpuBroadcastNdarray;
template<typename T, int, typename X>
class XpuTransposeNdarray;
template<typename T, int, typename X>
class XpuReshapeNdarray;

template<typename DerivedT, typename T>
class XpuNdarrayBase {
 public:
  XpuNdarrayBase() = default;
  ~XpuNdarrayBase() = default;

  template<template<typename> class unary_func>
  OF_DEVICE_FUNC XpuUnaryFuncNdarray<T, unary_func, DerivedT> UnaryFunc() const {
    return XpuUnaryFuncNdarray<T, unary_func, DerivedT>(*static_cast<const DerivedT*>(this));
  }
  template<template<typename> class binary_func, typename X>
  OF_DEVICE_FUNC XpuBinaryFuncNdarray<T, binary_func, DerivedT, X> BinaryFunc(const X& x) const {
    return XpuBinaryFuncNdarray<T, binary_func, DerivedT, X>(*static_cast<const DerivedT*>(this),
                                                             x);
  }
  OF_DEVICE_FUNC XpuBroadcastNdarray<const T> Broadcast(const XpuShape& shape) const {
    return XpuBroadcastNdarray<const T>(shape, *static_cast<const DerivedT*>(this));
  }
  template<int NDIMS>
  OF_DEVICE_FUNC XpuTransposeNdarray<T, NDIMS, DerivedT> Transpose(
      const int64_t perm[NDIMS]) const {
    return XpuTransposeNdarray<T, NDIMS, DerivedT>(*static_cast<const DerivedT*>(this), perm);
  }
  template<int NDIMS>
  OF_DEVICE_FUNC XpuReshapeNdarray<T, NDIMS, DerivedT> Reshape(const int64_t shape[NDIMS]) {
    return XpuReshapeNdarray<T, NDIMS, DerivedT>(*static_cast<const DerivedT*>(this), shape);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BASE_H_
