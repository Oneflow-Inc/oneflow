#ifndef ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct AdditiveAngularMarginKernelUtilImpl final {
  static void Forward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                      const T* in, const K* label, const int64_t lower_bound, const T cos_m,
                      const T sin_m, T* sin_theta_data, T* out);
  static void Backward(DeviceCtx* ctx, const int64_t batch_num, const int64_t labels_num,
                       const T* out_diff, const K* label, const int64_t lower_bound, const T cos_m,
                       const T sin_m, const T* sin_theta_data, T* in_diff);
};

template<DeviceType device_type, typename T>
struct AdditiveAngularMarginKernelUtil final {
  static void Forward(DeviceCtx* ctx, const Blob* in, const Blob* label, const int64_t lower_bound,
                      const T cos_m, const T sin_m, Blob* sin_theta_data, Blob* out);
  static void Backward(DeviceCtx* ctx, const Blob* out_diff, const int64_t lower_bound,
                       const T cos_m, const T sin_m, const Blob* label, const Blob* sin_theta_data,
                       Blob* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADDITIVE_ANGULAR_MARGIN_KERNEL_UTIL_H_
