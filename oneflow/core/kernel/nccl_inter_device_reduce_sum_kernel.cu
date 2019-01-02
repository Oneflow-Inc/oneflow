#include "oneflow/core/kernel/nccl_inter_device_reduce_sum_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

template<>
struct NcclInterDeviceReduceSumKernelUtil<DeviceType::kGPU> {
  static void ReduceSum(DeviceCtx* ctx, Blob* send, Blob* recv) {
    NcclUtil::AllReduce(ctx, send, recv);
  }
};

}  // namespace oneflow
