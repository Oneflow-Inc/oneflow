#ifndef ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct SparseCrossEntropyKernelUtil {
  static void ComputeEntropy(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                             const K* labels, T* y, const int64_t lower_bound = 0);
  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                          const K* labels, T* dx, const int64_t lower_bound = 0);
  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                          const K* labels, const T* dy, T* dx, const int64_t lower_bound = 0);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_
