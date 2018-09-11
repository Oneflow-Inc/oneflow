#ifndef ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

#ifdef WITH_CUDA

class NcclDeviceCtx final : public CudaDeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclDeviceCtx);
  NcclDeviceCtx(CudaStreamHandle* cuda_handler, ncclComm_t nccl_handler)
      : CudaDeviceCtx(cuda_handler), nccl_handler_(nccl_handler) {}
  ~NcclDeviceCtx() override = default;

  std::unique_ptr<DeviceCtx> Copy() const override {
    return std::unique_ptr<DeviceCtx>(new NcclDeviceCtx(cuda_handler_, nccl_handler_));
  }

  const ncclComm_t& nccl_handle() const override { return nccl_handler_; }

 private:
  ncclComm_t nccl_handler_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_DEVICE_CONTEXT_H_
