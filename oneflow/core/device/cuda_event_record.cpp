#include "oneflow/core/device/cuda_event_record.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

#ifdef WITH_CUDA

namespace {

int64_t GetCurrentDeviceId() {
  int device_id = -1;
  OF_CUDA_CHECK(cudaGetDevice(&device_id));
  CHECK_EQ(device_id, GlobalProcessCtx::Rank());
  return device_id;
}

}

CudaEventRecord::CudaEventRecord(DeviceCtx* device_ctx)
    : CudaEventRecord(GetCurrentDeviceId(), device_ctx) {}

CudaEventRecord::CudaEventRecord(int64_t device_id, DeviceCtx* device_ctx)
    : device_id_(device_id) {
  CudaCurrentDeviceGuard guard(device_id_);
  OF_CUDA_CHECK(
      cudaEventCreateWithFlags(&event_, cudaEventBlockingSync | cudaEventDisableTiming));
  OF_CUDA_CHECK(cudaEventRecord(event_, device_ctx->cuda_stream()));
}

bool CudaEventRecord::QueryDone() const {
  CudaCurrentDeviceGuard guard(device_id_);
  return cudaEventQuery(event_) == cudaSuccess;
}

#endif  // WITH_CUDA

}
