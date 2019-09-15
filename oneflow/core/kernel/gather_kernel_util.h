#ifndef ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct GatherKernelUtil final {
  static void Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out);
  static void Backward(DeviceCtx* ctx, const Blob* indices, const Blob* out_diff, int64_t axis,
                       Blob* in_diff);
  static void Forward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out,
                      const int64_t offset);
  static void Backward(DeviceCtx* ctx, const Blob* indices, const Blob* out_diff, int64_t axis,
                       Blob* in_diff, const int64_t offset);
};

template<DeviceType device_type, typename T, typename K>
struct GatherKernelUtilImpl final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, const int64_t offset);
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* out_diff,
                       const Shape& flat_in_shape, T* in_diff, const int64_t offset);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_GATHER_KERNEL_UTIL_H_
