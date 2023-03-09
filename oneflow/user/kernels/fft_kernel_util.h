#ifndef ONEFLOW_USER_KERNELS_FFT_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_FFT_KERNEL_UTIL_H_
#include "oneflow/core/ep/include/stream.h"

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"

#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/data_type.h"

#include "oneflow/core/kernel/util/numerics.cuh"
#include "oneflow/core/kernel/util/numeric_limits.cuh"
#include "oneflow/user/kernels/clip_by_value_kernel.h"

#include "oneflow/core/ndarray/xpu_util.h"

#include "oneflow/core/operator/operator_util.h"
#ifdef WITH_CUDA
#include "oneflow/core/cuda/atomic.cuh"
#endif  // WITH_CUDA

namespace oneflow{

template<DeviceType device_type, typename T, typename IDX>
struct FftKernelUtil{
    static void FftForward(ep::Stream* stream, const IDX elem_num, 
                           const T* x, const T* other, T* y);
};

// macros for functors instantiate(used by grid_sample_kernel_util.cu, grid_sample_kernel_util.cpp)
#define INSTANTIATE_FFT_KERNEL_UTIL(device_type, dtype_pair, itype_pair)  \
  template struct FftKernelUtil<device_type, OF_PP_PAIR_FIRST(dtype_pair), \
                                       OF_PP_PAIR_FIRST(itype_pair)>;



}   // namespace oneflow

#endif // ONEFLOW_USER_KERNELS_FFT_KERNEL_UTIL_H_