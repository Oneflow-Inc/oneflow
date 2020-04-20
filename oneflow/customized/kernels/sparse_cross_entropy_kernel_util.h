#ifndef ONEFLOW_CUSTOMIZED_KERNELS_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct SparseCrossEntropyKernelUtil {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const T* x, const K* labels, T* y);
  static void ComputeDiff(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const T* x, const K* labels, const T* dy, T* dx);
  static void BackwardSub(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const K* labels, const T* dy, T* dx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_
