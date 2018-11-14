#ifndef ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_ASSIGN_H_
#define ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_ASSIGN_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<int NDIMS, typename T, typename X>
OF_DEVICE_FUNC void CpuNdArrayAssign(XpuVarNdarray<T>* y, const X& x) {
  size_t n = y->shape().ElemNum();
  FOR_RANGE(int, i, 0, n) { *(y->template Mut<NDIMS>(i)) = x.template Get<NDIMS>(i); }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_CPU_NDARRAY_ASSIGN_H_
