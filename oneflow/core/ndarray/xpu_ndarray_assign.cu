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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename X, int NDIMS>
__global__ void NdarrayAssignReducedGpu(XpuVarNdarray<T> y,
                                        const XpuReducedNdarray<X, NDIMS> reduced) {
  NdarrayAssignCore<T, X, NDIMS>::Assign(y, reduced);
}

template<typename T, typename X, int NDIMS>
__global__ void NdarrayAssignGpu(XpuVarNdarray<T> y, const XpuVarNdarray<const X> x) {
  NdarrayAssignCore<T, X, NDIMS>::Assign(y, x);
}

}  // namespace

template<typename T, typename X, int NDIMS>
struct NdarrayAssignCoreWrapper<DeviceType::kCUDA, T, X, NDIMS> final {
  static void Assign(ep::Stream* stream, XpuVarNdarray<T>* y,
                     const XpuReducedNdarray<X, NDIMS>& reduced) {
    size_t n = y->host_shape().HostElemNum();
    RUN_CUDA_KERNEL((NdarrayAssignReducedGpu<T, X, NDIMS>), stream, n, *y, reduced);
  }
  static void Assign(ep::Stream* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const X>& x) {
    size_t n = y.host_shape().HostElemNum();
    if (n == 0) { return; }
    RUN_CUDA_KERNEL((NdarrayAssignGpu<T, X, NDIMS>), ctx, n, y, x);
  }
};

#define INSTANTIATE_NDARRAY_ASSIGN(ret_dtype_pair, dtype_pair, NDIMS)                           \
  template struct NdarrayAssignCoreWrapper<DeviceType::kCUDA, OF_PP_PAIR_FIRST(ret_dtype_pair), \
                                           OF_PP_PAIR_FIRST(dtype_pair), NDIMS>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
    INSTANTIATE_NDARRAY_ASSIGN,
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ,
    ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ, DIM_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_ASSIGN, HALF_DATA_TYPE_SEQ, HALF_DATA_TYPE_SEQ,
                                 DIM_SEQ);

}  // namespace oneflow
