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
#include "oneflow/core/ndarray/ndarray_apply_broadcast_binary_core.h"

namespace oneflow {

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinaryCoreWrapper<DeviceType::kCPU, T, NDIMS, binary_func> final {
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    NdarrayApplyBroadcastBinaryCore<T, NDIMS, binary_func>::Apply(y, a, b);
  }
};

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastInplaceBinaryCoreWrapper<DeviceType::kCPU, T, NDIMS, binary_func>
    final {
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    NdarrayApplyBroadcastBinaryCore<T, NDIMS, binary_func>::InplaceApply(y, x);
  }
};

#define INSTANTIATE_BROADCAST_BINARY_FUNC(dtype_pair, NDIMS, binary_func) \
  template struct NdarrayApplyBroadcastBinaryCoreWrapper<                 \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_BINARY_FUNC,
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, DIM_SEQ,
                                 BINARY_FUNC_SEQ);

#define INSTANTIATE_BROADCAST_INPLACE_BINARY_FUNC(dtype_pair, NDIMS, binary_func) \
  template struct NdarrayApplyBroadcastInplaceBinaryCoreWrapper<                  \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_INPLACE_BINARY_FUNC,
                                 ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, DIM_SEQ,
                                 ARITHMETIC_BINARY_FUNC_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_INPLACE_BINARY_FUNC,
                                 ((int8_t, DataType::kInt8)), DIM_SEQ, LOGICAL_BINARY_FUNC_SEQ);

}  // namespace oneflow
