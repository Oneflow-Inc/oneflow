#include "oneflow/core/kernel/multi_ring_all_reduce_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

template<typename T>
struct MultiRingAllReduceKernelUtil<DeviceType::kCPU, T> {
  static void Copy(DeviceCtx* ctx, T* dst, const T* src, int64_t size) { UNIMPLEMENTED(); }
  static void Copy(DeviceCtx* ctx, T* dst0, T* dst1, const T* src, int64_t size) {
    UNIMPLEMENTED();
  }
  static void Reduce(DeviceCtx* ctx, T* dst, const T* src0, const T* src1, int64_t size) {
    UNIMPLEMENTED();
  }
  static void Reduce(DeviceCtx* ctx, T* dst0, T* dst1, const T* src0, const T* src1, int64_t size) {
    UNIMPLEMENTED();
  }
};

#define INSTANTIATE_CPU_KERNEL_UTIL(type_cpp, type_proto) \
  template struct MultiRingAllReduceKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_CPU_KERNEL_UTIL

}  // namespace oneflow
