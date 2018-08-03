#ifndef ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

#ifdef WITH_CUDA

class CudaDeviceCtx final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaDeviceCtx);
  CudaDeviceCtx() = delete;
  ~CudaDeviceCtx() = default;

  CudaDeviceCtx(CudaStreamHandle* cuda_handler) : cuda_handler_(cuda_handler) {}
  std::unique_ptr<DeviceCtx> Copy() const {
    return std::unique_ptr<DeviceCtx>(new CudaDeviceCtx(cuda_handler_));
  }

  const cudaStream_t& cuda_stream() const { return *(cuda_handler_->cuda_stream()); }
  const cublasHandle_t& cublas_pmh_handle() const { return *(cuda_handler_->cublas_pmh_handle()); }
  const cublasHandle_t& cublas_pmd_handle() const { return *(cuda_handler_->cublas_pmd_handle()); }
  const cudnnHandle_t& cudnn_handle() const { return *(cuda_handler_->cudnn_handle()); }

  void AddCallBack(std::function<void()> callback) const override {
    cuda_handler_->AddCallBack(callback);
  }

 private:
  CudaStreamHandle* cuda_handler_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_
