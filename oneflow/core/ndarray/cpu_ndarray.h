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
#ifndef ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_H_

#include <climits>
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_shape.h"

namespace oneflow {

template<typename T, int NDIMS>
class CpuNdarray {
 public:
  using dtype = T;
  static const int ndims = NDIMS;

  ALWAYS_INLINE const XpuShape& xpu_shape() const { return xpu_shape_; }

 protected:
  explicit CpuNdarray(const Shape& shape) : xpu_shape_(shape) {}
  explicit CpuNdarray(const XpuShape& xpu_shape) : xpu_shape_(xpu_shape) {}
  virtual ~CpuNdarray() = default;

 private:
  XpuShape xpu_shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_H_
