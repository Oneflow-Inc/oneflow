#ifndef ONEFLOW_CORE_KERNEL_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename K, typename T, typename IDX>
struct IndexedSlicesReduceSumKernelUtil {
  static void ReduceSum(DeviceCtx* ctx, int64_t n, int64_t m, const K* indices, const T* values,
                        IDX* num_unique_indices, K* indices_out, T* values_out, void* workspace,
                        int64_t workspace_size_in_bytes);
  static void GetReduceSumWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n, int64_t m,
                                               int64_t* workspace_size_in_bytes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INDEXED_SLICES_REDUCE_SUM_KERNEL_UTIL_H_
