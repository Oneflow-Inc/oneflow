#ifndef ONEFLOW_CUSTOMIZED_KERNELS_DROPOUT_KERNEL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_DROPOUT_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>  //, typename U = void>
struct DropoutKernelUtil final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x,
                           const int8_t* mask, T* y);
};

template<DeviceType device_type>
struct RandomMaskLikeKernelUtil2 final {
  static void GenMask(DeviceCtx* ctx, const int64_t n, float threshold, const float* random_tmp,
                      int8_t* mask);
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_DROPOUT_KERNEL_UTIL_H_
