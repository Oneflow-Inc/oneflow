#ifndef ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_HELPER_H_
#define ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_HELPER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_reduce.h"
#include "oneflow/core/ndarray/xpu_ndarray_assign.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
class XpuNdArrayHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(XpuNdArrayHelper);
  XpuNdArrayHelper() = default;
  ~XpuNdArrayHelper() = default;

  void Reduce(DeviceCtx* ctx, XpuVarNdarray<T>* y, const XpuVarNdarray<const T>& x,
              XpuVarNdarray<T>* tmp_storage) {
    NdArrayReduce<device_type, T, NDIMS>::Reduce(ctx, y, x, tmp_storage);
    Assign(ctx, y, XpuReducedNdarray<T, NDIMS>(y->shape(), tmp_storage));
  }

  void Assign(DeviceCtx* ctx, XpuVarNdarray<T>* y, const XpuReducedNdarray<T, NDIMS>& reduced) {
    XpuNdArrayAssign<device_type, T, NDIMS>::Assign(ctx, y, reduced);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_HELPER_H_
