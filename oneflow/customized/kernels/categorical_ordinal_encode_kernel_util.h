#ifndef ONEFLOW_CUSTOMIZED_KERNELS_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct CategoricalOrdinalEncodeKernelUtil {
  static void Encode(DeviceCtx* ctx, int64_t capacity, T* table, T* size, int64_t n, const T* hash,
                     T* out);
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_H_
