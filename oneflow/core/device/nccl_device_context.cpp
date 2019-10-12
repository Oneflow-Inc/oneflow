#include "oneflow/core/device/nccl_device_context.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

#ifdef WITH_CUDA

NcclDeviceCtx::NcclDeviceCtx(ncclComm_t nccl_handler) : nccl_handler_(nccl_handler) {
  int32_t dev_id;
  NcclCheck(ncclCommCuDevice(nccl_handler_, &dev_id));
  {
    CudaCurrentDeviceGuard dev_guard(dev_id);
    CudaCheck(cudaStreamCreate(&cuda_stream_));
  }
  cpu_task_poller_ = std::thread([this, dev_id] {
    CudaCurrentDeviceGuard dev_guard(dev_id);
    std::function<void()> task;
    while (cpu_task_chan_.Receive(&task) == kChannelStatusSuccess) { task(); }
  });
  gpu_cb_event_poller_ = std::thread([this, dev_id]() {
    CudaCurrentDeviceGuard dev_guard(dev_id);
    CudaCBEvent cb_event;
    while (gpu_cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      CudaCheck(cudaEventSynchronize(cb_event.event));
      cb_event.callback();
      CudaCheck(cudaEventDestroy(cb_event.event));
    }
  });
}

NcclDeviceCtx::~NcclDeviceCtx() {
  cpu_task_chan_.Close();
  cpu_task_poller_.join();
  gpu_cb_event_chan_.Close();
  gpu_cb_event_poller_.join();
  CudaCheck(cudaStreamSynchronize(cuda_stream_));
  CudaCheck(cudaStreamDestroy(cuda_stream_));
}

void NcclDeviceCtx::Enqueue(const std::function<void()>& callback) const {
  CHECK_EQ(cpu_task_chan_.Send(callback), kChannelStatusSuccess);
}

void NcclDeviceCtx::AddCallBack(std::function<void()> callback) const {
  Enqueue([this, callback] {
    CudaCBEvent cb_event;
    CudaCheck(
        cudaEventCreateWithFlags(&cb_event.event, cudaEventBlockingSync | cudaEventDisableTiming));
    CudaCheck(cudaEventRecord(cb_event.event, cuda_stream_));
    cb_event.callback = callback;
    CHECK_EQ(gpu_cb_event_chan_.Send(cb_event), kChannelStatusSuccess);
  });
}

#endif  // WITH_CUDA

}  // namespace oneflow
