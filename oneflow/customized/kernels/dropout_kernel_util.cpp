#include "oneflow/core/common/preprocessor.h"
#include "oneflow/customized/kernels/dropout_kernel_util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<typename T>
struct DropoutKernelUtil<DeviceType::kCPU, T> final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float scale, const T* x,
                           const int8_t* mask, T* y) {
    for (int64_t i = 0; i < n; ++i) { y[i] = x[i] * static_cast<T>(mask[i]) * scale; }
  }
};

template<>
struct RandomMaskLikeKernelUtil2<DeviceType::kCPU> final {
  static void GenMask(DeviceCtx* ctx, const int64_t n, float threshold, const float* random_tmp,
                      int8_t* mask) {
    for (int64_t i = 0; i < n; ++i) { mask[i] = random_tmp[i] > threshold; }
  }
};

template struct RandomMaskLikeKernelUtil2<DeviceType::kCPU>;

#define INITIATE_DROPOUT_KERNEL_UTIL_CPU(T, type_proto) \
  template struct DropoutKernelUtil<DeviceType::kCPU, T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_DROPOUT_KERNEL_UTIL_CPU,
                     ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
#undef INITIATE_DROPOUT_KERNEL_UTIL_CPU

}  // namespace oneflow
