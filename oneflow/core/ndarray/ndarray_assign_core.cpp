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
#include "oneflow/core/ndarray/ndarray_assign_core.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T, typename X, int NDIMS>
struct NdarrayAssignCoreWrapper<DeviceType::kCPU, T, X, NDIMS> final {
  static void Assign(ep::Stream* stream, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<X, NDIMS>& reduced) {
    NdarrayAssignCore<T, X, NDIMS>::Assign(y, reduced);
  }
  static void Assign(ep::Stream* stream, const XpuVarNdarray<T>& y,
                     const XpuVarNdarray<const X>& x) {
    NdarrayAssignCore<T, X, NDIMS>::Assign(y, x);
  }
};

#define INSTANTIATE_NDARRAY_ASSIGN(ret_dtype_pair, dtype_pair, NDIMS)                          \
  template struct NdarrayAssignCoreWrapper<DeviceType::kCPU, OF_PP_PAIR_FIRST(ret_dtype_pair), \
                                           OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    INSTANTIATE_NDARRAY_ASSIGN,
    ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
    ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
    DIM_SEQ);

}  // namespace oneflow
