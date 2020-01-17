#include "oneflow/core/kernel/regular_gradient_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

template<typename T>
struct RegularGradientKernelUtil<DeviceType::kCPU, T> {
  static void RegularGradient(DeviceCtx* ctx, int64_t n, const T* model, const T* model_diff,
                              T* out, const T l1_scale, const T l2_scale) {
    FOR_RANGE(int64_t, i, 0, n) {
      const T model_val = model[i];
      out[i] = model_diff[i] + l1_scale * (model_val >= 0 ? 1 : -1) + l2_scale * model_val;
    }
  }
};

#define INSTANTIATE_REGULAR_GRADIENT_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct RegularGradientKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_REGULAR_GRADIENT_KERNEL_UTIL_CPU, FLOATING_DATA_TYPE_SEQ);
#undef INSTANTIATE_REGULAR_GRADIENT_KERNEL_UTIL_CPU

}  // namespace oneflow
