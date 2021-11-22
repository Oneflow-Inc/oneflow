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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_CORE_H_

#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_broadcast_ndarray.h"
#include "oneflow/core/ndarray/xpu_binary_func_ndarray.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinaryCoreWrapper final {
  static void Apply(ep::Stream* stream,
                    const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
                    const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b);
};

template<DeviceType device_type, typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastInplaceBinaryCoreWrapper final {
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y,
                           const XpuVarNdarray<const T>& x);
};

template<typename T, int NDIMS, template<typename> class binary_func>
struct NdarrayApplyBroadcastBinaryCore final {
  OF_DEVICE_FUNC static void Apply(
      const XpuVarNdarray<typename BinaryFuncTrait<binary_func, T>::return_type>& y,
      const XpuVarNdarray<const T>& a, const XpuVarNdarray<const T>& b) {
    const auto& ret =
        a.Broadcast(y.shape()).template BinaryFunc<binary_func>(b.Broadcast(y.shape()));
    y.template Assign<NDIMS>(ret);
  }
  OF_DEVICE_FUNC static void InplaceApply(const XpuVarNdarray<T>& y,
                                          const XpuVarNdarray<const T>& x) {
    y.template BinaryAssign<binary_func, NDIMS>(x.Broadcast(y.shape()));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_BROADCAST_BINARY_CORE_H_
