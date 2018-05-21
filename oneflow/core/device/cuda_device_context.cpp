#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/thread/gpu_thread.h"

namespace oneflow {

#ifdef WITH_CUDA

void CudaDeviceCtx::AddCallBack(std::function<void()> callback_stack) const {
  CudaEventCB event_cb;
  // Using cudaEventDisableTiming has better performance but cost more CPUs
  // CudaCheck(cudaEventCreateWithFlags(&(event_cb.event), cudaEventDisableTiming));
  CudaCheck(cudaEventCreateWithFlags(&(event_cb.event), cudaEventBlockingSync));
  CudaCheck(cudaEventRecord(event_cb.event, cuda_stream()));
  event_cb.callback = callback_stack;
  static_cast<GpuThread*>(cur_thread_)->AddEventCb(event_cb);
}

#endif  // WITH_CUDA

}  // namespace oneflow
