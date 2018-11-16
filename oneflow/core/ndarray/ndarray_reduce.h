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
    XpuVarNdarray<T> storage(x.shape(), tmp_storage.ptr());
    ExecShape cur_shape(x.shape());
    bool is_first_reduced_dim = true;
    CHECK_EQ(y.shape().NumAxes(), x.shape().NumAxes());
    if (x.shape() == y.shape()) {
      XpuNdArrayAssign<device_type, T, NDIMS>::Assign(ctx, y, x);
      return;
    }
    for (int i = 0; i < x.shape().NumAxes(); ++i) {
      if (y.shape().At(i) == x.shape().At(i)) { continue; }
      CHECK_EQ(y.shape().At(i), 1);
      CHECK_GT(x.shape().At(i), y.shape().At(i));
      if (is_first_reduced_dim) {
        ReduceAxis(ctx, i, storage, &cur_shape, x);
        is_first_reduced_dim = false;
      }
      ImplaceReduceAxis(ctx, i, storage, &cur_shape);
    }
    XpuReducedNdarray<T, NDIMS> reduced(y.shape(), storage);
    XpuNdArrayAssign<device_type, T, NDIMS>::Assign(ctx, y, reduced);
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    ExecShape cur_shape(x.shape());
    for (int i = 0; i < x.shape().NumAxes(); ++i) {
      if (y.shape().At(i) == x.shape().At(i)) { continue; }
      CHECK_EQ(y.shape().At(i), 1);
      CHECK_GT(x.shape().At(i), 1);
      ImplaceReduceAxis(ctx, i, x, &cur_shape);
    }
    XpuReducedNdarray<T, NDIMS> reduced(y.shape(), x);
    XpuNdArrayAssign<device_type, T, NDIMS>::Assign(ctx, y, reduced);
  }

  static void ReduceAxis(DeviceCtx* ctx, int axis, const XpuVarNdarray<T>& dst,
                         ExecShape* cur_shape, const XpuVarNdarray<const T>& x) {
    int64_t new_dim_value = cur_shape->At(axis) / 4;
    cur_shape->Set(axis, new_dim_value);
    XpuReducedNdarray<T, NDIMS> dst_reduced(*cur_shape, dst);
    NdArrayReduceCoreWrapper<device_type, T, NDIMS>::ReduceAxis(ctx, dst_reduced, x, axis);
  }

  static void ImplaceReduceAxis(DeviceCtx* ctx, int axis, const XpuVarNdarray<T>& implace,
                                ExecShape* cur_shape) {
    while (cur_shape->At(axis) > 1) {
      XpuReducedNdarray<T, NDIMS> from(*cur_shape, implace);
      int64_t new_dim_value = (cur_shape->At(axis) < 8 ? 1 : cur_shape->At(axis) / 4);
      cur_shape->Set(axis, new_dim_value);
      XpuReducedNdarray<T, NDIMS> to(*cur_shape, implace);
      NdArrayReduceCoreWrapper<device_type, T, NDIMS>::ReduceAxis(ctx, to, from, axis);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
