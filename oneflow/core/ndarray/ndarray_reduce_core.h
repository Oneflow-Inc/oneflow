#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_CORE_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"
#include "oneflow/core/ndarray/xpu_transpose_ndarray.h"
#include "oneflow/core/ndarray/xpu_reshape_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_ndarray_builder.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct NdArrayReduceCoreWrapper final {
  static void ReduceAxis(DeviceCtx* ctx, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuReducedNdarray<T, NDIMS>& x, int axis);
};

template<typename T, int NDIMS>
struct NdArrayReduceCore final {
  template<typename X>
  OF_DEVICE_FUNC static void ReduceAxis(const XpuReducedNdarray<T, NDIMS>& dst_reduced, const X& x,
                                        int axis) {
    size_t n = dst_reduced.shape().ElemNum();
    int64_t dst_dim_val = dst_reduced.shape().At(axis);
    XPU_1D_KERNEL_LOOP(i, n) {
      T* dst_reduced_ptr = dst_reduced.template Mut(i);
      int64_t coord[NDIMS];
      dst_reduced.shape().template Offset2Coordinate<NDIMS>(i, coord);
      T sum = 0;
      while (coord[axis] < x.shape().At(axis)) {
        sum += x.template Get<NDIMS>(coord);
        coord[axis] += dst_dim_val;
      }
      *dst_reduced_ptr = sum;
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_CORE_H_
