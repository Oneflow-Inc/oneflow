#ifndef ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

class DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCtx);
  virtual ~DeviceCtx() = default;

  virtual std::unique_ptr<DeviceCtx> Copy() const = 0;

#ifdef WITH_CUDA
  virtual const cudaStream_t& cuda_stream() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_pmh_handle() const { UNIMPLEMENTED(); }
  virtual const cublasHandle_t& cublas_pmd_handle() const { UNIMPLEMENTED(); }
  virtual const cudnnHandle_t& cudnn_handle() const { UNIMPLEMENTED(); }
  virtual const ncclComm_t& nccl_handle() const { UNIMPLEMENTED(); }
  virtual const ncclComm_t& nccl_scatter_handle() const { UNIMPLEMENTED(); }
  virtual const ncclComm_t& nccl_gather_handle() const { UNIMPLEMENTED(); }
  virtual const cudaStream_t& nccl_scatter_stream() const { UNIMPLEMENTED(); }
  virtual const cudaStream_t& nccl_gather_stream() const { UNIMPLEMENTED(); }
#endif

  virtual void AddCallBack(std::function<void()>) const = 0;

 protected:
  DeviceCtx() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_DEVICE_CONTEXT_H_
