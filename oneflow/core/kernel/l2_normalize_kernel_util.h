#ifndef ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct L2NormalizeKernelUtil final {
 public:
  static void Forward(DeviceCtx* ctx, const int32_t axis, const float epsilon, const Blob* in_blob,
                      Blob* square_x_sum_blob, Blob* out_blob);
  static void Backward(DeviceCtx* ctx, const int32_t axis, const float epsilon,
                       const Blob* out_blob, const Blob* out_diff_blob,
                       const Blob* square_x_sum_blob, Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_L2_NORMALIZE_KERNEL_UTIL_H_
