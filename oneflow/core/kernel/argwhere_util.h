#ifndef ONEFLOW_CORE_KERNEL_ARGWHERE_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ARGWHERE_UTIL_H_

#include "oneflow/core/device/device_context.h"

namespace oneflow {

template<typename T, typename I>
cudaError_t InferCubSelectFlaggedTempStorageBytes(DeviceCtx* ctx, int num_items, size_t& tmp_bytes);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ARGWHERE_UTIL_H_
