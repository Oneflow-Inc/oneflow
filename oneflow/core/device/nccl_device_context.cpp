#include "oneflow/core/device/nccl_device_context.h"

#ifdef WITH_CUDA

namespace oneflow {

NcclDeviceCtx::NcclDeviceCtx(ncclComm_t nccl_handle) : nccl_handle_(nccl_handle) {
  int32_t gpu_phy_id;
  NcclCheck(ncclCommCuDevice(nccl_handle_, &gpu_phy_id));
  cudaSetDevice(gpu_phy_id);
  cudaStreamCreate(&cuda_stream_);
  cpu_cb_event_poller_ = std::thread([this, gpu_phy_id]() {
    CudaCheck(cudaSetDevice(gpu_phy_id));
    std::function<void()> callback;
    while (cpu_cb_event_chan_.Receive(&callback) == kChannelStatusSuccess) { callback(); }
  });

  gpu_cb_event_poller_ = std::thread([this, gpu_phy_id]() {
    CudaCheck(cudaSetDevice(gpu_phy_id));
    CudaCBEvent cb_event;
    while (gpu_cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      CudaCheck(cudaEventSynchronize(cb_event.event));
      cb_event.callback();
      CudaCheck(cudaEventDestroy(cb_event.event));
    }
  });
}

NcclDeviceCtx::~NcclDeviceCtx() {
  cpu_cb_event_chan_.Close();
  cpu_cb_event_poller_.join();
  gpu_cb_event_chan_.Close();
  gpu_cb_event_poller_.join();
  cudaStreamSynchronize(cuda_stream_);
  cudaStreamDestroy(cuda_stream_);
}

void NcclDeviceCtx::AddCallBack(std::function<void()> callback) const {
  Enqueue([this, callback]() {
    CudaCBEvent cb_event;
    CudaCheck(
        cudaEventCreateWithFlags(&cb_event.event, cudaEventBlockingSync | cudaEventDisableTiming));
    cudaEventRecord(cb_event.event, cuda_stream_);
    cb_event.callback = callback;
    gpu_cb_event_chan_.Send(cb_event);
  });
}

void NcclDeviceCtx::Enqueue(const std::function<void()>& callback) const {
  cpu_cb_event_chan_.Send(callback);
}

}  // namespace oneflow

#endif  // WITH_CUDA