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
#include "oneflow/core/ndarray/ndarray_apply_binary_core.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T, template<typename> class binary_func>
struct NdarrayApplyBinaryCoreWrapper<DeviceType::kCPU, T, binary_func> final {
  static void Apply(ep::Stream* stream,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    NdarrayApplyBinaryCore<T, binary_func>::Apply(y.shape().ElemNum(), y.ptr(), a.ptr(), b.ptr());
  }
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    NdarrayApplyBinaryCore<T, binary_func>::InplaceApply(y.shape().ElemNum(), y.ptr(), x.ptr());
  }
};

#define INSTANTIATE_NDARRAY_APPLY_BINARY_CORE(dtype_pair, binary_func)                          \
  template struct NdarrayApplyBinaryCoreWrapper<DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), \
                                                binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_APPLY_BINARY_CORE,
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
                                     UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 ARITHMETIC_BINARY_FUNC_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_APPLY_BINARY_CORE,
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
                                     UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 LOGICAL_BINARY_FUNC_SEQ)

}  // namespace oneflow
