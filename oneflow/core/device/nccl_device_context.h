#ifndef ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cuda_device_context.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

#ifdef WITH_CUDA

class NcclDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclDeviceCtx);
  NcclDeviceCtx(ncclComm_t nccl_handler) : nccl_handle_(nccl_handler) {
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

  ~NcclDeviceCtx() override {
    cpu_cb_event_chan_.Close();
    cpu_cb_event_poller_.join();
    gpu_cb_event_chan_.Close();
    gpu_cb_event_poller_.join();
    cudaStreamSynchronize(cuda_stream_);
    cudaStreamDestroy(cuda_stream_);
  }

  const ncclComm_t& nccl_handle() const override { return nccl_handle_; }

  const cudaStream_t& cuda_stream() const override { return cuda_stream_; }

  void AddCallBack(std::function<void()> callback) const override {
    Enqueue([this, callback]() {
      CudaCBEvent cb_event;
      CudaCheck(cudaEventCreateWithFlags(&cb_event.event,
                                         cudaEventBlockingSync | cudaEventDisableTiming));
      cudaEventRecord(cb_event.event, cuda_stream_);
      cb_event.callback = callback;
      gpu_cb_event_chan_.Send(cb_event);
    });
  }

  void Enqueue(const std::function<void()>& callback) const { cpu_cb_event_chan_.Send(callback); }

 private:
  ncclComm_t nccl_handle_;
  cudaStream_t cuda_stream_;
  mutable Channel<std::function<void()>> cpu_cb_event_chan_;
  mutable Channel<CudaCBEvent> gpu_cb_event_chan_;
  std::thread cpu_cb_event_poller_;
  std::thread gpu_cb_event_poller_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_
