#ifndef ONEFLOW_CORE_KERNEL_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename K, typename V>
struct CategoricalHashTableLookupKernelUtil {
  static void GetOrInsert(DeviceCtx* ctx, int64_t capacity, K* keys, V* values, V* size, int64_t n,
                          const K* hash, V* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CATEGORICAL_HASH_TABLE_LOOKUP_KERNEL_UTIL_H_
