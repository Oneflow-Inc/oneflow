#ifndef _DEVICE_DEVICE_RESOURCE_H_
#define _DEVICE_DEVICE_RESOURCE_H_

#include <cstdint>
#include <string>
#include "device/device_alternate.h"

namespace caffe {
class Stream {
public:
  Stream();
  ~Stream();

  cudaStream_t get_cuda_stream() const { return stream_; }
private:
  cudaStream_t stream_;

  Stream(const Stream& other) = delete;
  Stream& operator=(const Stream& other) = delete;
};

class CublasHandle {
public:
  CublasHandle();
  ~CublasHandle();

  cublasHandle_t get_cublas_handle() const { return cublas_handle_; }
private:
  cublasHandle_t cublas_handle_;

  CublasHandle(const CublasHandle& other) = delete;
  CublasHandle& operator=(const CublasHandle& other) = delete;
};

class CudnnHandle {
public:
  CudnnHandle();
  ~CudnnHandle();

  cudnnHandle_t get_cudnn_handle() const { return cudnn_handle_; }
private:
  cudnnHandle_t cudnn_handle_;

  CudnnHandle(const CudnnHandle& other) = delete;
  CudnnHandle& operator=(const CudnnHandle& other) = delete;
};
}  // namespace caffe
#endif  // _DEVICE_DEVICE_RESOURCE_H_
