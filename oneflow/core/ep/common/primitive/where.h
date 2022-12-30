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
#ifndef ONEFLOW_CORE_EP_COMMON_PRIMITIVE_WHERE_H_
#define ONEFLOW_CORE_EP_COMMON_PRIMITIVE_WHERE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace where_details {

inline bool IsDimsEquals(size_t ndim, const int64_t* a_dims, const int64_t* b_dims) {
  for (size_t i = 0; i < ndim; ++i) {
    if (a_dims[i] != b_dims[i]) { return false; }
  }
  return true;
}

inline void GetCompactBroadcastDims(const size_t cond_ndim, const int64_t* cond_dims,
                                    const size_t x_ndim, const int64_t* x_dims, const size_t y_ndim,
                                    const int64_t* y_dims, size_t& compt_ndim,
                                    int64_t* cmpt_cond_dims, int64_t* cmpt_x_dims,
                                    int64_t* cmpt_y_dims, int64_t* cmpt_z_dims) {
  size_t max_ndim = std::max(x_ndim, y_ndim);
  max_ndim = std::max(max_ndim, cond_ndim);
  CHECK_LE(max_ndim, kMaxNumDims);

  auto MakeGetDimSize = [max_ndim](size_t ndim, const int64_t* dims) {
    size_t lpad = max_ndim - ndim;
    return [lpad, dims](int dim) -> int64_t { return dim < lpad ? 1 : dims[dim - lpad]; };
  };
  auto GetCondDimSize = MakeGetDimSize(cond_ndim, cond_dims);
  auto GetXDimSize = MakeGetDimSize(x_ndim, x_dims);
  auto GetYDimSize = MakeGetDimSize(y_ndim, y_dims);

  compt_ndim = 0;
  bool cond_pred_dim_broadcast = false;
  bool x_pred_dim_broadcast = false;
  bool y_pred_dim_broadcast = false;
  for (int i = 0; i < max_ndim; ++i) {
    int64_t cond_dim_size = GetCondDimSize(i);
    int64_t x_dim_size = GetXDimSize(i);
    int64_t y_dim_size = GetYDimSize(i);
    int64_t dim_size = std::max(std::max(x_dim_size, y_dim_size), cond_dim_size);
    if (dim_size == 1) { continue; }
    bool cond_broadcast = (cond_dim_size == 1);
    bool x_broadcast = (x_dim_size == 1);
    bool y_broadcast = (y_dim_size == 1);
    if (compt_ndim > 0 && cond_broadcast == cond_pred_dim_broadcast
        && x_broadcast == x_pred_dim_broadcast && y_broadcast == y_pred_dim_broadcast) {
      cmpt_cond_dims[compt_ndim - 1] *= cond_dim_size;
      cmpt_x_dims[compt_ndim - 1] *= x_dim_size;
      cmpt_y_dims[compt_ndim - 1] *= y_dim_size;
      cmpt_z_dims[compt_ndim - 1] *= dim_size;
    } else {
      cmpt_cond_dims[compt_ndim] = cond_dim_size;
      cmpt_x_dims[compt_ndim] = x_dim_size;
      cmpt_y_dims[compt_ndim] = y_dim_size;
      cmpt_z_dims[compt_ndim] = dim_size;
      compt_ndim += 1;
      cond_pred_dim_broadcast = cond_broadcast;
      x_pred_dim_broadcast = x_broadcast;
      y_pred_dim_broadcast = y_broadcast;
    }
  }
}

template<typename T, typename CondT>
struct WhereFunctor {
  OF_DEVICE_FUNC WhereFunctor() {}

  OF_DEVICE_FUNC T operator()(CondT cond, T x, T y) const { return cond ? x : y; }
};

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  OF_DEVICE_FUNC Pack() {}

  PackType<T, N> storage;
  T elem[N];
};

}  // namespace where_details

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_WHERE_H_
