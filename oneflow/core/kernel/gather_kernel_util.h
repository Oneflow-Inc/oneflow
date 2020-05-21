#ifndef ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct GatherKernelUtil final {
  static void Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out);
  static void Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out,
                      int64_t offset);
};

template<DeviceType device_type, typename T, typename K>
struct GatherKernelUtilImpl final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, int64_t offset);
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000
#define GATHER_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
#else
#define GATHER_DATA_TYPE_SEQ ARITHMETIC_DATA_TYPE_SEQ
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_
