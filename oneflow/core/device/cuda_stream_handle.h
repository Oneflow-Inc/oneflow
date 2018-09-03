#ifndef ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
#define ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA

struct CudaCBEvent {
  std::function<void()> callback;
  cudaEvent_t event;
};

class CudaStreamHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamHandle);
  CudaStreamHandle() = delete;
  CudaStreamHandle(Channel<CudaCBEvent>* cb_event_chan, int64_t dev_id)
      : cb_event_chan_(cb_event_chan), dev_id_(dev_id) {}

  const cudaStream_t* cuda_stream();
  const cublasHandle_t* cublas_pmh_handle();
  const cublasHandle_t* cublas_pmd_handle();
  const cudnnHandle_t* cudnn_handle();
  const ncclComm_t* nccl_handle();
  const ncclComm_t* nccl_scatter_handle();
  const ncclComm_t* nccl_gather_handle();

  void AddCallBack(std::function<void()> callback);

  ~CudaStreamHandle();

 private:
  Channel<CudaCBEvent>* cb_event_chan_;
  std::unique_ptr<cudaStream_t> cuda_stream_;
  std::unique_ptr<cublasHandle_t> cublas_pmh_handle_;
  std::unique_ptr<cublasHandle_t> cublas_pmd_handle_;
  std::unique_ptr<cudnnHandle_t> cudnn_handle_;
  std::unique_ptr<ncclComm_t> nccl_handle_;
  std::unique_ptr<ncclComm_t> nccl_scatter_handle_;
  std::unique_ptr<ncclComm_t> nccl_gather_handle_;
  int64_t dev_id_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
