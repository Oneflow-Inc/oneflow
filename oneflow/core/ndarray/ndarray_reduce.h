#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/ndarray_reduce_core.h"
#include "oneflow/core/ndarray/xpu_ndarray_assign.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct NdArrayReduce final {
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    ExecShape cur_shape(x.shape());
    for (int i = 0; i < x.shape().NumAxes(); ++i) {
      if (y.shape().At(i) == x.shape().At(i)) { continue; }
      CHECK_EQ(y.shape().At(i), 1);
      CHECK_GT(x.shape().At(i), 1);
      if (i == 0) { ReduceAxis(ctx, i, tmp_storage, &cur_shape, x); }
      ImplaceReduceAxis(ctx, i, tmp_storage, &cur_shape);
    }
    XpuReducedNdarray<T, NDIMS> reduced(y.shape(), tmp_storage);
    XpuNdArrayAssign<device_type, T, NDIMS>::Assign(ctx, y, reduced);
  }

  static void ReduceAxis(DeviceCtx* ctx, int axis, const XpuVarNdarray<T>& dst,
                         ExecShape* cur_shape, const XpuVarNdarray<const T>& x) {
    int64_t new_dim_value = cur_shape->At(axis) / 2;
    NdArrayReduceCoreWrapper<device_type, T, NDIMS>::ReduceAxis(ctx, dst.ptr(), x, axis,
                                                                new_dim_value);
    cur_shape->Set(axis, new_dim_value);
  }

  static void ImplaceReduceAxis(DeviceCtx* ctx, int axis, const XpuVarNdarray<T>& implace,
                                ExecShape* cur_shape) {
    while (cur_shape->At(axis) > 1) {
      const auto& reduced = XpuReducedNdarray<T, NDIMS>(*cur_shape, implace);
      int64_t new_dim_value = (cur_shape->At(axis) < 16 ? 1 : cur_shape->At(axis) / 4);
      NdArrayReduceCoreWrapper<device_type, T, NDIMS>::ImplaceReduceAxis(ctx, reduced, axis,
                                                                         new_dim_value);
      cur_shape->Set(axis, new_dim_value);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
