#ifndef ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_

#include "oneflow/core/ndarray/binary_func.h"

namespace oneflow {

template<typename T, const T (*binary_func)(const T, const T)>
class XpuBinaryFuncNdarray final {
 public:
  OF_DEVICE_FUNC XpuBinaryFuncNdarray(const XpuBroadcastNdarray<const T>& a_ndarray,
                                      const XpuBroadcastNdarray<const T>& b_ndarray)
      : a_ndarray_(&a_ndarray), b_ndarray_(&b_ndarray) {}

  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    return binary_func(a_ndarray_->Get<NDIMS>(offset), b_ndarray_->Get<NDIMS>(offset));
  }

 private:
  const XpuBroadcastNdarray<const T>* a_ndarray_;
  const XpuBroadcastNdarray<const T>* b_ndarray_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_
