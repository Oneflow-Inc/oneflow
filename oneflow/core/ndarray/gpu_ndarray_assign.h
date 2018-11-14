#ifndef ONEFLOW_CORE_NDARRAY_GPU_NDARRAY_ASSIGN_H_
#define ONEFLOW_CORE_NDARRAY_GPU_NDARRAY_ASSIGN_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

template<int NDIMS, typename T, typename X>
__device__ void GpuNdArrayAssign(XpuVarNdarray<T>* y, const X& x) {
  size_t n = y->shape().ElemNum();
  CUDA_1D_KERNEL_LOOP(i, n) { *(y->template Mut<NDIMS>(i)) = x.template Get<NDIMS>(i); }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_GPU_NDARRAY_ASSIGN_H_
