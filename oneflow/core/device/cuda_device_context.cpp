#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/thread/gpu_thread.h"

namespace oneflow {

#ifdef WITH_CUDA

void CudaDeviceCtx::AddCallBack(std::function<void()> callback_stack) const {
  CudaEventCB event_cb;
  CudaCheck(cudaEventCreateWithFlags(&(event_cb.event), cudaEventDisableTiming));
  event_cb.callback = callback_stack;
  static_cast<GpuThread*>(cur_thread_)->AddEventCb(event_cb);
}

#endif  // WITH_CUDA

}  // namespace oneflow
