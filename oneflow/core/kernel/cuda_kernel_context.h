#ifndef ONEFLOW_CORE_KERNEL_CUDA_KERNEL_CONTEXT_H_
#define ONEFLOW_CORE_KERNEL_CUDA_KERNEL_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

class CudaKernelCtx final : public KernelCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CudaKernelCtx);
  CudaKernelCtx() = delete;
  ~CudaKernelCtx() = default;

  CudaKernelCtx(const cudaStream_t* cuda_stream,
                const cublasHandle_t* cublas_handle) {
    set_cuda_stream(cuda_stream);
    set_cublas_handle(cublas_handle);
  }

  void AddCallBack(std::function<void()> callback) const override;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_CUDA_KERNEL_CONTEXT_H_
