#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

template<typename T>
struct NormalizationKernelUtil<kCPU, T> {
  static void ForwardTraining(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* beta,
                              Blob* y, Blob* moving_mean, Blob* moving_variance, Blob* mean,
                              Blob* inv_variance, Blob* buf, int32_t axis, double epsilon,
                              double momentum) {
    UNIMPLEMENTED();
  }
  static void ForwardInference(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* beta,
                               const Blob* moving_mean, const Blob* moving_variance, Blob* y,
                               Blob* buf, int32_t axis, double epsilon) {
    UNIMPLEMENTED();
  }
  static void Backward(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* mean,
                       const Blob* inv_variance, const Blob* dy, Blob* dx, Blob* gamma_diff,
                       Blob* beta_diff, Blob* buf, int32_t axis, double epsilon) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_NORMALIZATION_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct NormalizationKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_NORMALIZATION_KERNEL_UTIL_CPU, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_NORMALIZATION_KERNEL_UTIL_CPU

}  // namespace oneflow
