#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct NdArrayAssignCoreWrapper final {
  static void Assign(DeviceCtx* ctx, XpuVarNdarray<T>* y,
                     const XpuReducedNdarray<T, NDIMS>& reduced);
};

template<typename T, int NDIMS>
struct NdArrayAssignCore final {
  OF_DEVICE_FUNC static void Assign(XpuVarNdarray<T>* y,
                                    const XpuReducedNdarray<T, NDIMS>& reduced) {
    y->template Assign<NDIMS>(reduced);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_
