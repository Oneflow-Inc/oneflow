#ifndef ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_

#include "unsupported/Eigen/CXX11/Tensor"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(DeviceCtx);
  virtual ~DeviceCtx() = default;

  int64_t work_stream_id() const { return work_stream_id_; }

#ifdef WITH_CUDA
  const cudaStream_t& cuda_stream() const { return *cuda_stream_; }
  // cublas handle with POINTER_MODE_HOST(pmh)
  const cublasHandle_t& cublas_pmh_handle() const { return *cublas_pmh_handle_; }
  // cublas handle with POINTER_MODE_DEVICE(pmd)
  const cublasHandle_t& cublas_pmd_handle() const { return *cublas_pmd_handle_; }
  const cudnnHandle_t& cudnn_handle() const { return *cudnn_handle_; }
  const Eigen::GpuDevice& eigen_gpu_device() const { return *eigen_gpu_device_; }
#endif

  virtual void AddCallBack(std::function<void()>) const = 0;

 protected:
  DeviceCtx()
      : work_stream_id_(-1)
#ifdef WITH_CUDA
        ,
        cuda_stream_(nullptr),
        cublas_pmh_handle_(nullptr),
        cublas_pmd_handle_(nullptr),
        cudnn_handle_(nullptr),
        eigen_gpu_device_(nullptr)
#endif
  {
  }

  void set_work_stream_id(int64_t val) { work_stream_id_ = val; }

#ifdef WITH_CUDA
  void set_cuda_stream(const cudaStream_t* val) { cuda_stream_ = val; }
  void set_cublas_pmh_handle(const cublasHandle_t* val) { cublas_pmh_handle_ = val; }
  void set_cublas_pmd_handle(const cublasHandle_t* val) { cublas_pmd_handle_ = val; }
  void set_cudnn_handle(const cudnnHandle_t* val) { cudnn_handle_ = val; }
  void set_eigen_gpu_device(const Eigen::GpuDevice* val) { eigen_gpu_device_ = val; }
#endif

 private:
  int64_t work_stream_id_;
#ifdef WITH_CUDA
  const cudaStream_t* cuda_stream_;
  const cublasHandle_t* cublas_pmh_handle_;
  const cublasHandle_t* cublas_pmd_handle_;
  const cudnnHandle_t* cudnn_handle_;
  const Eigen::GpuDevice* eigen_gpu_device_;
#endif
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_
