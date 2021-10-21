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
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

#ifdef WITH_CUDA

namespace {

constexpr int kCudaEventReuseRecycleThreshold = 32;

}

CudaStreamHandle::CudaStreamHandle(Channel<CudaCBEvent>* cb_event_chan)
    : cb_event_chan_(cb_event_chan),
      cuda_stream_(nullptr),
      cublas_handle_(nullptr),
      cudnn_handle_(nullptr) {
  cuda_event_flags_ = cudaEventDisableTiming;
  if (ParseBooleanFromEnv("ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC", false)) {
    cuda_event_flags_ |= cudaEventBlockingSync;
  }
  reuse_cuda_event_ = ParseBooleanFromEnv("ONEFLOW_STREAM_REUSE_CUDA_EVENT", false);
}

cudaStream_t CudaStreamHandle::cuda_stream() {
  if (cuda_stream_ == nullptr) { OF_CUDA_CHECK(cudaStreamCreate(&cuda_stream_)); }
  return cuda_stream_;
}

cublasHandle_t CudaStreamHandle::cublas_handle() {
  if (cublas_handle_ == nullptr) {
    OF_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    OF_CUBLAS_CHECK(cublasSetStream(cublas_handle_, cuda_stream()));
#if CUDA_VERSION >= 11000
    if (Global<ResourceDesc, ForSession>::Get()->enable_tensor_float_32_compute()) {
      OF_CUBLAS_CHECK(cublasSetMathMode(cublas_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
    }
#endif
  }
  return cublas_handle_;
}

cudnnHandle_t CudaStreamHandle::cudnn_handle() {
  if (cudnn_handle_ == nullptr) {
    if (IsCuda9OnTuringDevice()) {
      OF_CUDA_CHECK(cudaDeviceSynchronize());
      OF_CUDA_CHECK(cudaGetLastError());
    }
    OF_CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    if (IsCuda9OnTuringDevice()) {
      OF_CUDA_CHECK(cudaDeviceSynchronize());
      cudaGetLastError();
    }
    OF_CUDNN_CHECK(cudnnSetStream(cudnn_handle_, cuda_stream()));
  }
  return cudnn_handle_;
}

void CudaStreamHandle::AddCallBack(std::function<void()> callback) {
  CudaCBEvent cb_event;
  cb_event.callback = std::move(callback);
  cudaEvent_t cuda_event;
  if (reuse_cuda_event_) {
    if (consumer_event_queue_.empty()) {
      std::unique_lock<std::mutex> lock(global_event_queue_mutex_);
      consumer_event_queue_.swap(global_event_queue_);
    }
    if (consumer_event_queue_.empty()) {
      OF_CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event, cuda_event_flags_));
    } else {
      cuda_event = consumer_event_queue_.back();
      consumer_event_queue_.pop_back();
    }
  } else {
    OF_CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event, cuda_event_flags_));
  }
  cb_event.event = cuda_event;
  OF_CUDA_CHECK(cudaEventRecord(cb_event.event, cuda_stream()));
  cb_event_chan_->Send(std::move(cb_event));
}

void CudaStreamHandle::SyncRecycleEvent(cudaEvent_t event) {
  OF_CUDA_CHECK(cudaEventSynchronize(event));
  if (reuse_cuda_event_) {
    producer_event_queue_.push_back(event);
    if (producer_event_queue_.size() >= kCudaEventReuseRecycleThreshold) {
      std::unique_lock<std::mutex> lock(global_event_queue_mutex_);
      global_event_queue_.insert(global_event_queue_.end(), producer_event_queue_.begin(),
                                 producer_event_queue_.end());
      producer_event_queue_.clear();
    }
  } else {
    OF_CUDA_CHECK(cudaEventDestroy(event));
  }
}

CudaStreamHandle::~CudaStreamHandle() {
  if (cuda_stream_ != nullptr) { OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream_)); }
  for (cudaEvent_t event : consumer_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
  for (cudaEvent_t event : producer_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
  for (cudaEvent_t event : global_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
  if (cudnn_handle_ != nullptr) { OF_CUDNN_CHECK(cudnnDestroy(cudnn_handle_)); }
  if (cublas_handle_ != nullptr) { OF_CUBLAS_CHECK(cublasDestroy(cublas_handle_)); }
  if (cuda_stream_ != nullptr) { OF_CUDA_CHECK(cudaStreamDestroy(cuda_stream_)); }
}

#endif  // WITH_CUDA

}  // namespace oneflow
