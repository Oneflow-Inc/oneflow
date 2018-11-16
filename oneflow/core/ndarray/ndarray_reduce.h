#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/ndarray_reduce_xpu.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct NdArrayReduce final {
  static void Reduce(DeviceCtx* ctx, XpuVarNdarray<T>* y, const XpuVarNdarray<const T>& x,
                     XpuVarNdarray<T>* tmp_storage) {
    ExecShape cur_shape(x.shape());
    for (int i = 0; i < x.shape().NumAxes(); ++i) {
      if (y->shape().At(i) == x.shape().At(i)) { continue; }
      CHECK_EQ(y->shape().At(i), 1);
      CHECK_GT(x->shape().At(i), 1);
      if (i == 0) { ReduceAxis(ctx, i, tmp_storage, &cur_shape, x); }
      ImplaceReduceAxis(ctx, i, tmp_storage, &cur_shape);
    }
  }

  static void ReduceAxis(DeviceCtx* ctx, int axis, XpuVarNdarray<T>* dst, ExecShape* cur_shape,
                         const XpuVarNdarray<const T>& x) {
    int64_t new_dim_value = cur_shape->At(axis) / 2;
    NdArrayReduceXpuWrapper<device_type, T, NDIMS>::ReduceAxis(ctx, dst->mut_ptr(), x, axis,
                                                               new_dim_value);
    cur_shape->Set(axis, new_dim_value);
  }

  static void ImplaceReduceAxis(DeviceCtx* ctx, int axis, XpuVarNdarray<T>* implace,
                                ExecShape* cur_shape) {
    while (cur_shape->At(axis) > 1) {
      const auto& reduced = XpuReducedNdarray<T, NDIMS>(*cur_shape, *implace);
      int64_t new_dim_value = (cur_shape->At(axis) < 16 ? 1 : cur_shape->At(axis) / 4);
      NdArrayReduceXpuWrapper<device_type, T, NDIMS>::ImplaceReduceAxis(ctx, reduced, axis,
                                                                        new_dim_value);
      cur_shape->Set(axis, new_dim_value);
    }
  }
};

}  // namespace oneflow

#endif ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_H_
