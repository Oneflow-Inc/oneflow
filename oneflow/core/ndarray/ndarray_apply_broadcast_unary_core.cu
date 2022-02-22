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
#include "oneflow/core/ndarray/ndarray_apply_broadcast_unary_core.h"

namespace oneflow {

namespace {

template<typename T, int NDIMS, template<typename> class unary_func>
__global__ void GpuBroadcastUnaryFunc(const XpuVarNdarray<T> y, const XpuVarNdarray<const T> x) {
  NdarrayApplyBroadcastUnaryCore<T, NDIMS, unary_func>::Apply(y, x);
}

}  // namespace

template<typename T, int NDIMS, template<typename> class unary_func>
struct NdarrayApplyBroadcastUnaryCoreWrapper<DeviceType::kCUDA, T, NDIMS, unary_func> final {
  static void Apply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                    const XpuVarNdarray<const T>& x) {
    size_t n = y.host_shape().HostElemNum();
    if (n == 0) { return; }
    RUN_CUDA_KERNEL((GpuBroadcastUnaryFunc<T, NDIMS, unary_func>), stream, n, y, x);
  }
};

#define INSTANTIATE_BROADCAST_UNARY_FUNC(dtype_pair, NDIMS, unary_func) \
  template struct NdarrayApplyBroadcastUnaryCoreWrapper<                \
      DeviceType::kCUDA, OF_PP_PAIR_FIRST(dtype_pair), NDIMS, unary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_BROADCAST_UNARY_FUNC,
                                 ARITHMETIC_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
                                 DIM_SEQ, ARITHMETIC_UNARY_FUNC_SEQ)
}  // namespace oneflow
