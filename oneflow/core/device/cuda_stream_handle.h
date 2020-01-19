#ifndef ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
#define ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA

struct CudaCBEvent {
  std::function<void()> callback;
  cudaEvent_t event;
  std::string op_name;
};

class CudaStreamHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamHandle);
  CudaStreamHandle() = delete;
  CudaStreamHandle(Channel<CudaCBEvent>* cb_event_chan) : cb_event_chan_(cb_event_chan) {}

  const cudaStream_t* cuda_stream();
  const cublasHandle_t* cublas_pmh_handle();
  const cublasHandle_t* cublas_pmd_handle();
  const cublasHandle_t* cublas_tensor_op_math_handle();
  const cudnnHandle_t* cudnn_handle();

  void AddCallBack(std::function<void()> callback);
  void AddCallBack(std::function<void()> callback, const std::string& op_name);

  ~CudaStreamHandle();

 private:
  Channel<CudaCBEvent>* cb_event_chan_;
  std::unique_ptr<cudaStream_t> cuda_stream_;
  std::unique_ptr<cublasHandle_t> cublas_pmh_handle_;
  std::unique_ptr<cublasHandle_t> cublas_pmd_handle_;
  std::unique_ptr<cublasHandle_t> cublas_tensor_op_math_handle_;
  std::unique_ptr<cudnnHandle_t> cudnn_handle_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
