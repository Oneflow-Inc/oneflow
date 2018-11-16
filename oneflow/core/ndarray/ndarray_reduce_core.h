#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_XPU_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_XPU_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct NdArrayReduceCoreWrapper final {
  static void ReduceAxis(DeviceCtx* ctx, T* dst_ptr, const XpuVarNdarray<const T>& x, int axis,
                         int64_t new_dim_value);
  static void ImplaceReduceAxis(DeviceCtx* ctx, const XpuReducedNdarray<T, NDIMS>& x, int axis,
                                int64_t new_dim_value);
};

template<typename T, int NDIMS>
struct NdArrayReduceCore final {
  OF_DEVICE_FUNC static void ReduceAxis(T* dst_ptr, const XpuVarNdarray<const T>& x, int axis,
                                        int64_t new_dim_value) {
    XpuVarNdarray<T> dst_var(x.shape(), dst_ptr);
    ExecShape to_shape(x.shape());
    to_shape.Set(axis, new_dim_value);
    XpuReducedNdarray<T, NDIMS> dst_reduced(to_shape, dst_var);
    XPU_1D_KERNEL_LOOP(i, to_shape.ElemNum()) {
      int64_t coord[NDIMS];
      ExecShapeUtil<NDIMS>::Offset2DimVec(to_shape, i, coord);
      T* dst_reduced_ptr = dst_reduced.template Mut(coord);
      T sum = 0;
      while (coord[axis] < x.shape().At(axis)) {
        sum += x.template Get<NDIMS>(coord);
        coord[axis] += new_dim_value;
      }
      *dst_reduced_ptr = sum;
    }
  }
  OF_DEVICE_FUNC static void ImplaceReduceAxis(const XpuReducedNdarray<T, NDIMS>& x, int axis,
                                               int64_t new_dim_value) {
    ExecShape to_shape(x.shape());
    to_shape.Set(axis, new_dim_value);
    XPU_1D_KERNEL_LOOP(i, to_shape.ElemNum()) {
      int64_t coord[NDIMS];
      ExecShapeUtil<NDIMS>::Offset2DimVec(to_shape, i, coord);
      T* dst_reduced_ptr = x.template Mut(coord);
      coord[axis] += new_dim_value;
      while (coord[axis] < x.shape().At(axis)) {
        *dst_reduced_ptr += x.template Get(coord);
        coord[axis] += new_dim_value;
      }
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_XPU_H_
