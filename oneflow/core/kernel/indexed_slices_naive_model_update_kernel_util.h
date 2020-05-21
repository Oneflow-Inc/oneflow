#ifndef ONEFLOW_CORE_KERNEL_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct IndexedSlicesNaiveMdUpdateKernelUtil final {
  static void Update(DeviceCtx* ctx, const K* indices, const T* values, const float* learning_rate,
                     int64_t num_indices, int64_t num_features, int64_t feature_size,
                     int64_t feature_id_offset, T* model);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_UTIL_H_
