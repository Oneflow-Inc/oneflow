/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
  CudaStreamHandle(Channel<CudaCBEvent>* cb_event_chan);

  cudaStream_t cuda_stream();
  cublasHandle_t cublas_handle();
  cudnnHandle_t cudnn_handle();

  void AddCallBack(std::function<void()> callback);
  void SyncRecycleEvent(cudaEvent_t event);

  ~CudaStreamHandle();

 private:
  Channel<CudaCBEvent>* cb_event_chan_;
  cudaStream_t cuda_stream_;
  cublasHandle_t cublas_handle_;
  cudnnHandle_t cudnn_handle_;
  int cuda_event_flags_;
  bool reuse_cuda_event_;
  std::deque<cudaEvent_t> consumer_event_queue_;
  std::deque<cudaEvent_t> producer_event_queue_;
  std::deque<cudaEvent_t> global_event_queue_;
  std::mutex global_event_queue_mutex_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_STREAM_HANDLE_H_
