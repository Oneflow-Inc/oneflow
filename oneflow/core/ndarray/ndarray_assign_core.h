#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct NdarrayAssignCoreWrapper final {
  static void Assign(DeviceCtx* ctx, const XpuVarNdarray<T>& y,
                     const XpuReducedNdarray<T, NDIMS>& reduced);
};

template<typename T, int NDIMS>
struct NdarrayAssignCore final {
  OF_DEVICE_FUNC static void Assign(const XpuVarNdarray<T>& y,
                                    const XpuReducedNdarray<T, NDIMS>& reduced) {
    y.template Assign<NDIMS>(reduced);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_CORE_H_
