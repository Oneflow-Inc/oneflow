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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct NdarrayAssignCoreWrapper final {
  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<T, NDIMS>& reduced);
};

template<typename T, int NDIMS>
struct NdarrayAssignCore final {
  OF_DEVICE_FUNC static void Assign(const XpuVarNdarray<T>& y,
                                    const XpuReducedNdarray<T, NDIMS>& reduced) {
    y.template Assign<NDIMS>(reduced);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_
