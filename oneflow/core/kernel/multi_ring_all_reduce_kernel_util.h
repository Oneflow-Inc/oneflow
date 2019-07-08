#ifndef ONEFLOW_CORE_KERNEL_MULTI_RING_ALL_REDUCE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_MULTI_RING_ALL_REDUCE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct MultiRingAllReduceKernelUtil {
  static void Copy(DeviceCtx* ctx, T* dst, const T* src, int64_t size);
  static void Copy(DeviceCtx* ctx, T* dst0, T* dst1, const T* src, int64_t size);
  static void Reduce(DeviceCtx* ctx, T* dst, const T* src0, const T* src1, int64_t size);
  static void Reduce(DeviceCtx* ctx, T* dst0, T* dst1, const T* src0, const T* src1, int64_t size);
};

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_KERNEL_MULTI_RING_ALL_REDUCE_KERNEL_UTIL_H_
