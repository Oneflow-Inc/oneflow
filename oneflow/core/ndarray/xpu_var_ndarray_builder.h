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
#ifndef ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_BUILDER_H_
#define ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_BUILDER_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

template<typename T>
class XpuVarNdarrayBuilder final {
 public:
  XpuVarNdarrayBuilder() = default;
  XpuVarNdarrayBuilder(const XpuVarNdarrayBuilder&) = default;
  ~XpuVarNdarrayBuilder() = default;

  XpuVarNdarray<T> operator()(const Shape& shape, T* ptr) const {
    return XpuVarNdarray<T>(shape, ptr);
  }
  template<typename DT = T>
  typename std::enable_if<!std::is_same<DT, const DT>::value, XpuVarNdarray<DT>>::type operator()(
      Blob* blob, int ndims_extend_to) const {
    return XpuVarNdarray<DT>(blob, ndims_extend_to);
  }
  template<typename DT = T>
  typename std::enable_if<!std::is_same<DT, const DT>::value, XpuVarNdarray<const DT>>::type
  operator()(const Blob* blob, int ndims_extend_to) const {
    return XpuVarNdarray<const DT>(blob, ndims_extend_to);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_BUILDER_H_
