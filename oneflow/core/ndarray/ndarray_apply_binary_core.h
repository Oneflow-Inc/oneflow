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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_CORE_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_binary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayApplyBinaryCoreWrapper final {
  static void Apply(ep::Stream* stream,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b);
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x);
};

template<typename T, template<typename> class binary_func>
struct NdarrayApplyBinaryCore final {
  OF_DEVICE_FUNC static void Apply(size_t n,
                                   typename BinaryFuncTrait<binary_func, T>::return_type* y,
                                   const T* a, const T* b) {
    XPU_1D_KERNEL_LOOP_BEGIN(i, n);
    y[i] = binary_func<T>::Invoke(a[i], b[i]);
    XPU_1D_KERNEL_LOOP_END();
  }
  OF_DEVICE_FUNC static void InplaceApply(size_t n, T* y, const T* x) {
    XPU_1D_KERNEL_LOOP_BEGIN(i, n);
    y[i] = binary_func<T>::Invoke(y[i], x[i]);
    XPU_1D_KERNEL_LOOP_END();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BINARY_CORE_H_
