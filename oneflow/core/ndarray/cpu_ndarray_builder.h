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
#ifndef ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_HELPER_H_
#define ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_HELPER_H_

#include "oneflow/core/ndarray/cpu_var_ndarray.h"
#include "oneflow/core/ndarray/cpu_slice_var_ndarray.h"
#include "oneflow/core/ndarray/cpu_concat_var_ndarray.h"

namespace oneflow {

template<typename default_data_type, int default_ndims>
class CpuNdarrayBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuNdarrayBuilder);
  CpuNdarrayBuilder() = default;
  ~CpuNdarrayBuilder() = default;

  template<typename T = default_data_type, int NDIMS = default_ndims>
  CpuVarNdarray<T, NDIMS> Var(const Shape& shape, T* ptr) const {
    return CpuVarNdarray<T, NDIMS>(shape, ptr);
  }
  template<typename T = default_data_type, int NDIMS = default_ndims>
  CpuVarNdarray<T, NDIMS> Var(const ShapeView& shape_view, T* ptr) const {
    return CpuVarNdarray<T, NDIMS>(shape_view, ptr);
  }
  template<int CONCAT_AXES = 0, typename T = default_data_type, int NDIMS = default_ndims>
  CpuConcatVarNdarray<T, NDIMS, CONCAT_AXES> Concatenate(
      const std::vector<CpuVarNdarray<T, NDIMS>>& var_ndarrays) const {
    return CpuConcatVarNdarray<T, NDIMS, CONCAT_AXES>(var_ndarrays);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_HELPER_H_
