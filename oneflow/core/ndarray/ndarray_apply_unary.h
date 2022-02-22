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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/ndarray_apply_unary_core.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class unary_func,
         typename Enable = void>
struct NdarrayApplyUnary;

template<DeviceType device_type, typename T, template<typename> class unary_func>
struct NdarrayApplyUnary<
    device_type, T, unary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y) {
    NdarrayApplyUnaryCoreWrapper<device_type, T, unary_func>::InplaceApply(stream, y);
  }
};

template<DeviceType device_type, typename T, template<typename> class unary_func>
struct NdarrayApplyUnary<
    device_type, T, unary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void InplaceApply(ep::Stream* stream, const XpuVarNdarray<T>& y) {
    using NewT = typename DevDType<device_type, T>::type;
    return NdarrayApplyUnary<device_type, NewT, unary_func>::InplaceApply(
        stream, reinterpret_cast<const XpuVarNdarray<NewT>&>(y));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_APPLY_UNARY_H_
