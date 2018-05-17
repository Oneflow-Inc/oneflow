#ifndef ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_

#include "unsupported/Eigen/CXX11/Tensor"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCtx);
  DeviceCtx() = delete;
  virtual ~DeviceCtx() = default;

  int64_t work_stream_id() const { return work_stream_id_; }
  void* buf_ptr() const { return buf_ptr_; }

#ifdef WITH_CUDA
  virtual const cudaStream_t& cuda_stream() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_pmh_handle() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_pmd_handle() const { UNIMPLEMENTED(); }
  virtual const cudnnHandle_t& cudnn_handle() const { UNIMPLEMENTED(); }
  virtual const Eigen::GpuDevice& eigen_gpu_device() const { UNIMPLEMENTED(); }
#endif

  virtual void AddCallBack(std::function<void()>) const = 0;

 protected:
  DeviceCtx(int64_t work_stream_id, void* buf_ptr)
      : work_stream_id_(work_stream_id), buf_ptr_(buf_ptr) {}

 private:
  int64_t work_stream_id_;
  void* buf_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_
