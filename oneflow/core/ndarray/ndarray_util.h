#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#if defined(__CUDACC__)
#define XPU_1D_KERNEL_LOOP(i, n) CUDA_1D_KERNEL_LOOP(i, n)
#else
#define XPU_1D_KERNEL_LOOP(i, n) FOR_RANGE(int64_t, i, 0, n)
#endif

#if defined(__CUDACC__)
OF_DEVICE_FUNC inline void XpuSyncThreads() { __syncthreads(); }
#else
inline void XpuSyncThreads() {}
#endif

#if defined(__CUDACC__)
#define OF_GLOBAL_FUNC __global__
#else
#define OF_GLOBAL_FUNC
#endif

#define GET_SEQ(n) OF_PP_CAT(OF_PP_CAT(GET_SEQ_, n), )
#define GET_SEQ_0 OF_PP_MAKE_TUPLE_SEQ(0)
#define GET_SEQ_1 GET_SEQ_0 OF_PP_MAKE_TUPLE_SEQ(1)
#define GET_SEQ_2 GET_SEQ_1 OF_PP_MAKE_TUPLE_SEQ(2)
#define GET_SEQ_3 GET_SEQ_2 OF_PP_MAKE_TUPLE_SEQ(3)
#define GET_SEQ_4 GET_SEQ_3 OF_PP_MAKE_TUPLE_SEQ(4)
#define GET_SEQ_5 GET_SEQ_5 OF_PP_MAKE_TUPLE_SEQ(5)
}

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_UTIL_H_
