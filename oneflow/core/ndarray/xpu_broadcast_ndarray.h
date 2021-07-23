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
#ifndef ONEFLOW_CORE_NDARRAY_XPU_BROADCAST_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_BROADCAST_NDARRAY_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_ndarray_base.h"

namespace oneflow {

template<typename T, int NDIMS>
struct XpuBroadcastNdarrayUtil;

template<typename T>
class XpuBroadcastNdarray final : public XpuNdarrayBase<XpuBroadcastNdarray<T>, T> {
 public:
  OF_DEVICE_FUNC XpuBroadcastNdarray(const XpuShape& shape, const XpuVarNdarray<T>& var)
      : shape_(shape), var_(var) {}
  OF_DEVICE_FUNC ~XpuBroadcastNdarray() = default;

  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    int64_t coord[NDIMS];
    shape_.template Offset2Coordinate<NDIMS>(offset, coord);
    XpuBroadcastNdarrayUtil<T, NDIMS>::SrcCoordinate(var_.shape(), coord);
    return var_.template Get<NDIMS>(coord);
  }

  OF_DEVICE_FUNC const XpuShape& shape() const { return shape_; }
  OF_DEVICE_FUNC const XpuVarNdarray<T>& var() const { return var_; }

 private:
  const XpuShape shape_;
  const XpuVarNdarray<T> var_;
};

#define IMPLACE_SET_SRC_COORD(i) coord[i] %= src_shape.At(i);
#define SPECIALIZE_XPU_BROADCAST_NDARRAY_UTIL(n)                                                \
  template<typename T>                                                                          \
  struct XpuBroadcastNdarrayUtil<T, n + 1> final {                                              \
    OF_DEVICE_FUNC static void SrcCoordinate(const XpuShape& src_shape, int64_t coord[n + 1]) { \
      OF_PP_FOR_EACH_TUPLE(IMPLACE_SET_SRC_COORD, GET_SEQ(n));                                  \
    }                                                                                           \
  }
SPECIALIZE_XPU_BROADCAST_NDARRAY_UTIL(0);
SPECIALIZE_XPU_BROADCAST_NDARRAY_UTIL(1);
SPECIALIZE_XPU_BROADCAST_NDARRAY_UTIL(2);
SPECIALIZE_XPU_BROADCAST_NDARRAY_UTIL(3);
SPECIALIZE_XPU_BROADCAST_NDARRAY_UTIL(4);
SPECIALIZE_XPU_BROADCAST_NDARRAY_UTIL(5);
#undef SPECIALIZE_XPU_BROADCAST_NDARRAY_UTIL
#undef IMPLACE_SET_SRC_COORD

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_BROADCAST_NDARRAY_H_
