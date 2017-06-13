#ifndef ONEFLOW_CORE_COMMON_CUDA_STREAM_HANDLE_H_
#define ONEFLOW_CORE_COMMON_CUDA_STREAM_HANDLE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class CudaStreamHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamHandle);
  CudaStreamHandle() = default;

  const cudaStream_t* cuda_stream() {
    if (!cuda_stream_) {
      cuda_stream_.reset(new cudaStream_t);
      CHECK_EQ(cudaStreamCreate(cuda_stream_.get()), 0);
    }
    return cuda_stream_.get();
  }

  const cublasHandle_t* cublas_handle() {
    if (!cublas_handle_) {
      cublas_handle_.reset(new cublasHandle_t);
      CHECK_EQ(cublasCreate(cublas_handle_.get()), 0);
      CHECK_EQ(cublasSetStream(*cublas_handle_, *cuda_stream()), 0);
    }
    return cublas_handle_.get();
  }

  const cudnnHandle_t* cudnn_handle() {
    if (!cudnn_handle_) {
      cudnn_handle_.reset(new cudnnHandle_t);
      CHECK_EQ(cudnnCreate(cudnn_handle_.get()), 0);
      CHECK_EQ(cudnnSetStream(*cudnn_handle_, *cuda_stream()), 0);
    }
    return cudnn_handle_.get();
  }

  ~CudaStreamHandle() {
    if (cudnn_handle_) {
      CHECK_EQ(cudnnDestroy(*cudnn_handle_), 0);
    }
    if (cublas_handle_) {
      CHECK_EQ(cublasDestroy(*cublas_handle_), 0);
    }
    if (cuda_stream_) {
      CHECK_EQ(cudaStreamDestroy(*cuda_stream_), 0);
    }
  }

 private:
  std::unique_ptr<cudaStream_t> cuda_stream_;
  std::unique_ptr<cublasHandle_t> cublas_handle_;
  std::unique_ptr<cudnnHandle_t> cudnn_handle_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_COMMON_CUDA_STREAM_HANDLE_H_
