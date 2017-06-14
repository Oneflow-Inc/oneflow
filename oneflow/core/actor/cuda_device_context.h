#ifndef ONEFLOW_CORE_ACTOR_CUDA_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_ACTOR_CUDA_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CudaDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CudaDeviceCtx);
  CudaDeviceCtx() = delete;
  ~CudaDeviceCtx() = default;

  CudaDeviceCtx(const cudaStream_t* cuda_stream,
                const cublasHandle_t* cublas_handle,
                const cudnnHandle_t* cudnn_handle) {
    set_cuda_stream(cuda_stream);
    set_cublas_handle(cublas_handle);
    set_cudnn_handle(cudnn_handle);
  }

  void AddCallBack(std::function<void()> callback) const override;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_CUDA_DEVICE_CONTEXT_H_
