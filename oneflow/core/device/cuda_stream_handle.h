#ifndef ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
#define ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA

class CudaStreamHandle;

struct CudaCBEvent {
  std::function<void()> callback;
  cudaEvent_t event;
  CudaStreamHandle* cuda_stream_handle;
};

class CudaStreamHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamHandle);
  CudaStreamHandle() = delete;
  CudaStreamHandle(Channel<CudaCBEvent>* cb_event_chan) : cb_event_chan_(cb_event_chan) {}
  ~CudaStreamHandle();

  const cudaStream_t* cuda_stream();
  const cublasHandle_t* cublas_pmh_handle();
  const cublasHandle_t* cublas_pmd_handle();
  const cudnnHandle_t* cudnn_handle();

  void AddCallBack(std::function<void()> callback);
  cudaEvent_t GetCudaEvent();
  void PutCudaEvent(cudaEvent_t event);

 private:
  Channel<CudaCBEvent>* cb_event_chan_;
  std::unique_ptr<cudaStream_t> cuda_stream_;
  std::unique_ptr<cublasHandle_t> cublas_pmh_handle_;
  std::unique_ptr<cublasHandle_t> cublas_pmd_handle_;
  std::unique_ptr<cudnnHandle_t> cudnn_handle_;
  std::deque<cudaEvent_t> cuda_event_pool_;
  std::atomic_flag cuda_event_pool_mutex_ = ATOMIC_FLAG_INIT;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
