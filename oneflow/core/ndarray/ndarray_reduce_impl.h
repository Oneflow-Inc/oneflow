#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct NdarrayMatrixRowReduce final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x);
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage);
};

template<DeviceType device_type, typename T>
struct NdarrayMatrixColReduce final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x);
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_
