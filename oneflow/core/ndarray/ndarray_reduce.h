#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/ndarray_reduce_impl.h"

namespace oneflow {

template<DeviceType device_type, typename T, const T (*binary_func)(const T, const T)>
struct NdArrayReduce final {
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK_EQ(y.shape().NumAxes(), x.shape().NumAxes());
    if (NdarrayNoReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayNoReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    } else if (NdarrayScalarReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayScalarReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    } else if (NdarrayMatrixRowReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayMatrixRowReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    } else if (NdarrayMatrixColReduce<device_type, T, binary_func>::Matched(y, x)) {
      NdarrayMatrixColReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    } else {
      NdarrayDefaultReduce<device_type, T, binary_func>::Reduce(ctx, y, x, tmp_storage);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
