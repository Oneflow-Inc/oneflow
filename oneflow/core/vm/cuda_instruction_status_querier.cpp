#include "oneflow/core/vm/cuda_instruction_status_querier.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {
namespace vm {

bool CudaInstrStatusQuerier::event_completed() const {
  cudaSetDevice(device_id_);
  return cudaEventQuery(event_) == cudaSuccess;
}

void CudaInstrStatusQuerier::SetLaunched(DeviceCtx* device_ctx) {
  cudaSetDevice(device_id_);
  CudaCheck(cudaEventCreateWithFlags(&event_, cudaEventBlockingSync | cudaEventDisableTiming));
  CudaCheck(cudaEventRecord(event_, device_ctx->cuda_stream()));
  launched_ = true;
}

}  // namespace vm
}  // namespace oneflow
