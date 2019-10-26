#ifndef ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct AdditiveAngularMarginKernelUtilImpl final {
  static void Forward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                      const int64_t lower_bound, const T cos_m, const T sin_m, const T* in,
                      const K* label, T* sin_theta_data, T* out);
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const int64_t lower_bound, const T cos_m, const T sin_m, const T* out_diff,
                       const K* label, const T* sin_theta_data, T* in_diff);
};

template<DeviceType device_type, typename T>
struct AdditiveAngularMarginKernelUtil final {
  static void Forward(DeviceCtx* ctx, const int64_t lower_bound, const T cos_m, const T sin_m,
                      const Blob* in, const Blob* label, Blob* sin_theta_data, Blob* out);
  static void Backward(DeviceCtx* ctx, const int64_t lower_bound, const T cos_m, const T sin_m,
                       const Blob* out_diff, const Blob* label, const Blob* sin_theta_data,
                       Blob* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_H_
