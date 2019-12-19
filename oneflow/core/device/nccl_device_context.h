#ifndef ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

#ifdef WITH_CUDA

class NcclDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclDeviceCtx);
  explicit NcclDeviceCtx(ncclComm_t nccl_handler);
  ~NcclDeviceCtx() override;

  const ncclComm_t& nccl_handle() const override { return nccl_handler_; }
  const cudaStream_t& cuda_stream() const override { return cuda_stream_; }
  void AddCallBack(std::function<void()> callback) const override;
  void Enqueue(const std::function<void()>& callback) const;

 private:
  ncclComm_t nccl_handler_;
  cudaStream_t cuda_stream_;
  mutable Channel<std::function<void()>> cpu_task_chan_;
  mutable Channel<CudaCBEvent> gpu_cb_event_chan_;
  std::thread cpu_task_poller_;
  std::thread gpu_cb_event_poller_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_
