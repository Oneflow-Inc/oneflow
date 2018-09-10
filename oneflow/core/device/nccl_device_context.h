#ifndef ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

#ifdef WITH_CUDA

class NcclDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclDeviceCtx);
  NcclDeviceCtx() = delete;
  ~NcclDeviceCtx() override = default;

  NcclDeviceCtx(CudaStreamHandle* cuda_handler, ncclComm_t nccl_handler)
      : cuda_handler_(cuda_handler), nccl_handler_(nccl_handler) {}
  std::unique_ptr<DeviceCtx> Copy() const override {
    return std::unique_ptr<DeviceCtx>(new NcclDeviceCtx(cuda_handler_, nccl_handler_));
  }

  const cudaStream_t& cuda_stream() const override { return *(cuda_handler_->cuda_stream()); }

  void AddCallBack(std::function<void()> callback) const override {
    cuda_handler_->AddCallBack(callback);
  }

  const ncclComm_t& nccl_handle() const override { return nccl_handler_; }

 private:
  CudaStreamHandle* cuda_handler_;
  ncclComm_t nccl_handler_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_
