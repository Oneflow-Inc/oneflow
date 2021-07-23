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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_unary_func_ndarray.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class unary_func>
struct NdarrayApplyUnaryCoreWrapper final {
  static void InplaceApply(DeviceCtx* ctx, const XpuVarNdarray<T>& y);
};

template<typename T, template<typename> class unary_func>
struct NdarrayApplyUnaryCore final {
  OF_DEVICE_FUNC static void InplaceApply(T* y, size_t n) {
    XPU_1D_KERNEL_LOOP(i, n) { y[i] = unary_func<T>::Invoke(y[i]); }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_CORE_H_
