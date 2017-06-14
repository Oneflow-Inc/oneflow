#include "oneflow/core/common/cuda_stream_handle.h"

namespace oneflow {

const cudaStream_t* CudaStreamHandle::cuda_stream() {
  if (!cuda_stream_) {
    cuda_stream_.reset(new cudaStream_t);
    CHECK_EQ(cudaStreamCreate(cuda_stream_.get()), 0);
  }
  return cuda_stream_.get();
}

const cublasHandle_t* CudaStreamHandle::cublas_handle() {
  if (!cublas_handle_) {
    cublas_handle_.reset(new cublasHandle_t);
    CHECK_EQ(cublasCreate(cublas_handle_.get()), 0);
    CHECK_EQ(cublasSetStream(*cublas_handle_, *cuda_stream()), 0);
  }
  return cublas_handle_.get();
}

const cudnnHandle_t* CudaStreamHandle::cudnn_handle() {
  if (!cudnn_handle_) {
    cudnn_handle_.reset(new cudnnHandle_t);
    CHECK_EQ(cudnnCreate(cudnn_handle_.get()), 0);
    CHECK_EQ(cudnnSetStream(*cudnn_handle_, *cuda_stream()), 0);
  }
  return cudnn_handle_.get();
}

CudaStreamHandle::~CudaStreamHandle() {
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

} // namespace oneflow
