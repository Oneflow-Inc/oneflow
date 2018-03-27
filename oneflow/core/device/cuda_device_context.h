#ifndef ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

#ifdef WITH_CUDA

class CudaDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CudaDeviceCtx);
  CudaDeviceCtx() = delete;
  ~CudaDeviceCtx() = default;

  CudaDeviceCtx(int64_t work_stream_id, const cudaStream_t* cuda_stream,
                const cublasHandle_t* cublas_pmh_handle = nullptr,
                const cublasHandle_t* cublas_pmd_handle = nullptr,
                const cudnnHandle_t* cudnn_handle = nullptr,
                const Eigen::GpuDevice* eigen_gpu_device = nullptr

  ) {
    set_work_stream_id(work_stream_id);
    set_cuda_stream(cuda_stream);
    set_cublas_pmh_handle(cublas_pmh_handle);
    set_cublas_pmd_handle(cublas_pmd_handle);
    set_cudnn_handle(cudnn_handle);
    set_eigen_gpu_device(eigen_gpu_device);
  }

  void AddCallBack(std::function<void()> callback) const override;

 private:
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_DEVICE_CONTEXT_H_
