#include "device/device_resource.h"
#include "device/device_alternate.h"

namespace caffe {
Stream::Stream() {
  CUDA_CHECK(cudaStreamCreate(&stream_));
}
Stream::~Stream() {
  CUDA_CHECK(cudaStreamDestroy(stream_));
}

CublasHandle::CublasHandle() {
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUBLAS_CHECK(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
}
CublasHandle::~CublasHandle() {
  CUBLAS_CHECK(cublasDestroy(cublas_handle_));
}

CudnnHandle::CudnnHandle() {
  CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
}
CudnnHandle::~CudnnHandle() {
  CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
}
}  // namespace caffe
