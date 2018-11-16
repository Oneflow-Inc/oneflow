#ifndef ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_
#define ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_

#include "oneflow/core/ndarray/xpu_ndarray_assign_xpu.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct XpuNdArrayAssign final {
  static void Assign(DeviceCtx* ctx, XpuVarNdarray<T>* y,
                     const XpuReducedNdarray<T, NDIMS>& reduced) {
    NdArrayAssignXpuWrapper<device_type, T, NDIMS>::Assign(ctx, y, reduced);
  }
};

}  // namespace oneflow

#endif ONEFLOW_CORE_NDARRAY_XPU_ASSIGN_H_
