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
#ifndef ONEFLOW_CORE_NDARRAY_XPU_REDUCED_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_REDUCED_NDARRAY_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/ndarray/unary_func.h"

namespace oneflow {

template<typename T, int NDIMS, typename X = XpuVarNdarray<T>>
class XpuReducedNdarray final {
 public:
  OF_DEVICE_FUNC XpuReducedNdarray(const XpuShape& shape, const X& data)
      : shape_(shape), data_(data) {}

  OF_DEVICE_FUNC const XpuShape& shape() const { return shape_; }
  const XpuShape& host_shape() const { return shape_; }
  OF_DEVICE_FUNC const X& data() const { return data_; }

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    int64_t coord[NDIMS];
    shape_.template Offset2Coordinate<NDIMS>(offset, coord);
    return Get(coord);
  }

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T Get(int64_t coord[ndims]) const {
    return data_.template Get<ndims>(coord);
  }

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t offset) const {
    int64_t coord[NDIMS];
    shape_.template Offset2Coordinate<NDIMS>(offset, coord);
    return Mut(coord);
  }

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t coord[NDIMS]) const {
    return data_.template Mut<NDIMS>(coord);
  }

 private:
  XpuShape shape_;
  X data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_REDUCED_NDARRAY_H_
