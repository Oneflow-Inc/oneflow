#ifndef ONEFLOW_CUSTOMIZED_KERNELS_SOFTMAX_KERNEL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_SOFTMAX_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct SoftmaxKernelUtil {
  static void ComputeProb(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* in, T* tmp,
                          T* prob, void* temp_storage, const size_t temp_storage_bytes);
  static void ComputeDiff(DeviceCtx* ctx, const int64_t n, const int64_t w, const T* dy,
                          const T* out, T* sum_vec, T* dx, void* temp_storage,
                          const size_t temp_storage_bytes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_SOFTMAX_KERNEL_UTIL_H_
