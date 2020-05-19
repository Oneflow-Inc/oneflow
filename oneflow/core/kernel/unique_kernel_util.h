#ifndef ONEFLOW_CORE_KERNEL_UNIQUE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_UNIQUE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename KEY, typename IDX>
struct UniqueKernelUtil {
  static void Unique(DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique, KEY* unique_out,
                     IDX* idx_out, void* workspace, int64_t workspace_size_in_bytes);
  static void UniqueWithCounts(DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique,
                               KEY* unique_out, IDX* idx_out, IDX* count, void* workspace,
                               int64_t workspace_size_in_bytes);
  static void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                            int64_t* workspace_size_in_bytes);
  static void GetUniqueWithCountsWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                                      int64_t* workspace_size_in_bytes);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UNIQUE_KERNEL_UTIL_H_
