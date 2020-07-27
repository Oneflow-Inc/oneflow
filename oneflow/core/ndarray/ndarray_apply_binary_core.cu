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

namespace {

template<typename T, template<typename> class binary_func>
__global__ void NdarrayApplyBinaryApplyGpu(size_t n,
                                           typename BinaryFuncTrait<binary_func, T>::return_type* y,
                                           const T* a, const T* b) {
  NdarrayApplyBinaryCore<T, binary_func>::Apply(n, y, a, b);
}

template<typename T, template<typename> class binary_func>
__global__ void NdarrayApplyBinaryInplaceApplyGpu(size_t n, T* y, const T* x) {
  NdarrayApplyBinaryCore<T, binary_func>::InplaceApply(n, y, x);
}

}  // namespace

template<typename T, template<typename> class binary_func>
struct NdarrayApplyBinaryCoreWrapper<DeviceType::kGPU, T, binary_func> final {
  static void Apply(DeviceCtx* ctx,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    size_t n = y.host_shape().HostElemNum();
    RUN_CUDA_KERNEL((NdarrayApplyBinaryApplyGpu<T, binary_func>), ctx, n, n, y.host_ptr(),
                    a.host_ptr(), b.host_ptr());
  }
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x) {
    size_t n = y.host_shape().HostElemNum();
    RUN_CUDA_KERNEL((NdarrayApplyBinaryInplaceApplyGpu<T, binary_func>), ctx, n, n, y.host_ptr(),
                    x.host_ptr());
  }
};

#define INSTANTIATE_NDARRAY_APPLY_BINARY_CORE(dtype_pair, binary_func)                          \
  template struct NdarrayApplyBinaryCoreWrapper<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair), \
                                                binary_func>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_NDARRAY_APPLY_BINARY_CORE,
                                 ARITHMETIC_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, BINARY_FUNC_SEQ);

}  // namespace oneflow
