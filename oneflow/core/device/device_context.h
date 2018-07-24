#ifndef ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCtx);
  DeviceCtx() = delete;
  virtual ~DeviceCtx() = default;

  void* buf_ptr() const { return buf_ptr_; }
  size_t buf_size() const { return buf_size_; }
  void set_buf_ptr(void* buf_ptr) { buf_ptr_ = buf_ptr; }
  void set_buf_size(size_t buf_size) { buf_size_ = buf_size; }
  virtual std::unique_ptr<DeviceCtx> Copy() const = 0;

#ifdef WITH_CUDA
  virtual const cudaStream_t& cuda_stream() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_pmh_handle() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_pmd_handle() const { UNIMPLEMENTED(); }
  virtual const cudnnHandle_t& cudnn_handle() const { UNIMPLEMENTED(); }
#endif

  virtual void AddCallBack(std::function<void()>) const = 0;

 protected:
  DeviceCtx(void* buf_ptr, size_t buf_size) : buf_ptr_(buf_ptr), buf_size_(buf_size) {}

 private:
  void* buf_ptr_;
  size_t buf_size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_
