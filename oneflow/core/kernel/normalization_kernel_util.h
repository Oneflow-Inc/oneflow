#ifndef ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct NormalizationKernelUtil {
  static void ForwardTraining(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* beta,
                              Blob* y, Blob* moving_mean, Blob* moving_variance, Blob* mean,
                              Blob* inv_variance, Blob* buf, int32_t axis, double epsilon,
                              double momentum);
  static void ForwardInference(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* beta,
                               const Blob* moving_mean, const Blob* moving_variance, Blob* out,
                               Blob* buf, int32_t axis, double epsilon);
  static void Backward(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* mean,
                       const Blob* inv_variance, const Blob* dy, Blob* dx, Blob* gamma_diff,
                       Blob* beta_diff, Blob* buf, int32_t axis, double epsilon);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_UTIL_H_
