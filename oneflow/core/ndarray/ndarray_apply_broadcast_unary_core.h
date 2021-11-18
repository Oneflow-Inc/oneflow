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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_CORE_H_

#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_broadcast_ndarray.h"
#include "oneflow/core/ndarray/xpu_unary_func_ndarray.h"
#include "oneflow/core/ndarray/unary_func.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS, template<typename> class unary_func>
struct NdarrayApplyBroadcastUnaryCoreWrapper final {
  static void Apply(ep::Stream* stream, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x);
};

template<typename T, int NDIMS, template<typename> class unary_func>
struct NdarrayApplyBroadcastUnaryCore final {
  OF_DEVICE_FUNC static void Apply(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    y.template Assign<NDIMS>(x.Broadcast(y.shape()).template UnaryFunc<unary_func>());
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_UNARY_CORE_H_
