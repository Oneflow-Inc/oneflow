#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/upsample_nearest_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

template<typename T>
struct UpsampleNearestUtil<DeviceType::kGPU, T> {
  static void Forward(const KernelCtx& ctx, const UpsampleNearestOpConf& conf, const Blob* in_blob,
                      Blob* out_blob) {}

  static void Backward(const KernelCtx& ctx, const UpsampleNearestOpConf& conf,
                       const Blob* out_diff_blob, Blob* in_diff_blob) {}
};

#define INSTANTIATE_UPSAMPLE_NEAREST_KERNEL_UTIL(type_cpp, type_proto) \
  template class UpsampleNearestUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_UPSAMPLE_NEAREST_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
