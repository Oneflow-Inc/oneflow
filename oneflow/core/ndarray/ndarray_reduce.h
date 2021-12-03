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
#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ndarray/ndarray_reduce_impl.h"

namespace oneflow {

template<DeviceType device_type, typename T, template<typename> class binary_func,
         typename Enable = void>
struct NdarrayReduce;

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayReduce<
    device_type, T, binary_func,
    typename std::enable_if<std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  using RetT = typename BinaryFuncTrait<binary_func, T>::return_type;
  static void Reduce(ep::Stream* stream, const XpuVarNdarray<RetT>& origin_y,
                     const XpuVarNdarray<const T>& origin_x, const XpuVarNdarray<T>& tmp_storage) {
    DimVector simplified_x_dim;
    DimVector simplified_y_dim;
    TrySimplifyDims(origin_x.shape(), origin_y.shape(), &simplified_x_dim, &simplified_y_dim);
    XpuVarNdarray<RetT> y(Shape(simplified_y_dim), origin_y.ptr());
    XpuVarNdarray<const T> x(Shape(simplified_x_dim), origin_x.ptr());

    CHECK_EQ(y.shape().NumAxes(), x.shape().NumAxes());
    if (NdarrayNoReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayNoReduce<device_type, T, binary_func>::Reduce(stream, y, x, tmp_storage);
    } else if (NdarrayScalarReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayScalarReduce<device_type, T, binary_func>::Reduce(stream, y, x, tmp_storage);
    } else if (NdarrayMatrixRowReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayMatrixRowReduce<device_type, T, binary_func>::Reduce(stream, y, x, tmp_storage);
    } else if (NdarrayMatrixColReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayMatrixColReduce<device_type, T, binary_func>::Reduce(stream, y, x, tmp_storage);
    } else if (NdarrayXYZCubeXZReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayXYZCubeXZReduce<device_type, T, binary_func>::Reduce(stream, y, x, tmp_storage);
    } else {
      NdarrayDefaultReduce<device_type, T, binary_func>::Reduce(stream, y, x, tmp_storage);
    }
  }

  static void TrySimplifyDims(const XpuShape& x, const XpuShape& y, DimVector* simplified_x,
                              DimVector* simplified_y) {
    CHECK_EQ(y.NumAxes(), x.NumAxes());
    CHECK(y.At(0) == 1 || y.At(0) == x.At(0));
    CHECK(simplified_x->empty());
    CHECK(simplified_y->empty());
    simplified_x->emplace_back(x.At(0));
    simplified_y->emplace_back(y.At(0));
    bool prev_axis_is_reduced = (y.At(0) == 1);
    FOR_RANGE(int, i, 1, x.NumAxes()) {
      const int64_t x_dim = x.At(i);
      const int64_t y_dim = y.At(i);
      const bool cur_axis_is_reduced = (y_dim == 1);
      CHECK(cur_axis_is_reduced || y_dim == x_dim);
      if (cur_axis_is_reduced == prev_axis_is_reduced) {
        simplified_x->back() *= x_dim;
        simplified_y->back() *= y_dim;
      } else {
        simplified_x->emplace_back(x_dim);
        simplified_y->emplace_back(y_dim);
      }
      prev_axis_is_reduced = cur_axis_is_reduced;
    }
  }
};

template<DeviceType device_type, typename T, template<typename> class binary_func>
struct NdarrayReduce<
    device_type, T, binary_func,
    typename std::enable_if<!std::is_same<T, typename DevDType<device_type, T>::type>::value>::type>
    final {
  static void Reduce(ep::Stream* stream, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    using NewT = typename DevDType<device_type, T>::type;
    return NdarrayReduce<device_type, NewT, binary_func>::Reduce(
        stream, reinterpret_cast<const XpuVarNdarray<NewT>&>(y),
        reinterpret_cast<const XpuVarNdarray<const NewT>&>(x),
        reinterpret_cast<const XpuVarNdarray<NewT>&>(tmp_storage));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
