#ifndef ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
#define ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class CudaStreamHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamHandle);
  CudaStreamHandle() = default;

  const cudaStream_t* cuda_stream();
  const cublasHandle_t* cublas_handle();
  const cudnnHandle_t* cudnn_handle();

  ~CudaStreamHandle();

 private:
  std::unique_ptr<cudaStream_t> cuda_stream_;
  std::unique_ptr<cublasHandle_t> cublas_handle_;
  std::unique_ptr<cudnnHandle_t> cudnn_handle_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
