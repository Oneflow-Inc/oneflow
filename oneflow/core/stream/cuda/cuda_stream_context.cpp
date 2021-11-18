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
#include "oneflow/core/stream/include/stream_context.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/cuda_check_numerics_kernel_observer.h"
#include "oneflow/core/graph/stream_id.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/channel.h"

#ifdef WITH_CUDA
#include <cublas_v2.h>

namespace oneflow {

namespace {

constexpr int kCudaEventReuseRecycleThreshold = 32;

class CudaStreamContext : public StreamContext, public KernelObserverProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamContext);
  explicit CudaStreamContext(int device_ordinal);
  virtual ~CudaStreamContext();

  Maybe<void> AddCallback(std::function<void()> callback) override;
  DeviceType device_type() const override { return DeviceType::kGPU; }
  KernelObserver* GetKernelObserver() override;

  ep::Stream* stream() override;

 private:
  cudaEvent_t GetEvent();
  void SyncRecycleEvent(cudaEvent_t event);

  ep::CudaStream stream_;
  Channel<std::pair<cudaEvent_t, std::function<void()>>> cb_event_chan_;
  int cuda_event_flags_;
  bool reuse_cuda_event_;
  std::deque<cudaEvent_t> consumer_event_queue_;
  std::deque<cudaEvent_t> producer_event_queue_;
  std::deque<cudaEvent_t> global_event_queue_;
  std::mutex global_event_queue_mutex_;
  std::thread poller_thread_;
  int device_ordinal_;
  std::unique_ptr<KernelObserver> kernel_observer_;
};

}  // namespace

CudaStreamContext::CudaStreamContext(int device_ordinal)
    : stream_(device_ordinal), device_ordinal_(device_ordinal) {
  CudaCurrentDeviceGuard guard(device_ordinal_);
  cuda_event_flags_ = cudaEventDisableTiming;
  if (ParseBooleanFromEnv("ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC", false)) {
    cuda_event_flags_ |= cudaEventBlockingSync;
  }
  reuse_cuda_event_ = ParseBooleanFromEnv("ONEFLOW_STREAM_REUSE_CUDA_EVENT", false);

  std::vector<std::shared_ptr<KernelObserver>> kernel_observers;
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS", false)) {
    LOG(WARNING) << "Environment variable ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS has been set "
                    "to a truthy "
                    "value, it will impact performance";
    kernel_observers.emplace_back(new CudaCheckNumericsKernelObserver());
  }
  kernel_observer_.reset(new ChainKernelObserver(kernel_observers));

  poller_thread_ = std::thread([this]() {
    CudaCurrentDeviceGuard guard(device_ordinal_);
    stream_.OnExecutionContextSetup();
    OF_PROFILER_NAME_THIS_HOST_THREAD("GPU " + std::to_string(device_ordinal_) + " Poller : ("
                                      + std::to_string(device_ordinal_) + ")");
    std::pair<cudaEvent_t, std::function<void()>> cb_event;
    while (cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      SyncRecycleEvent(cb_event.first);
      cb_event.second();
    }
    stream_.OnExecutionContextTeardown();
  });
}

CudaStreamContext::~CudaStreamContext() {
  CudaCurrentDeviceGuard guard(device_ordinal_);
  cb_event_chan_.Close();
  poller_thread_.join();
  for (cudaEvent_t event : consumer_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
  for (cudaEvent_t event : producer_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
  for (cudaEvent_t event : global_event_queue_) { OF_CUDA_CHECK(cudaEventDestroy(event)); }
}

Maybe<void> CudaStreamContext::AddCallback(std::function<void()> callback) {
  cudaEvent_t cuda_event = GetEvent();
  OF_CUDA_CHECK(cudaEventRecord(cuda_event, stream_.cuda_stream()));
  cb_event_chan_.Send(std::make_pair(cuda_event, std::move(callback)));
  return Maybe<void>::Ok();
}

cudaEvent_t CudaStreamContext::GetEvent() {
  cudaEvent_t cuda_event{};
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
  return cuda_event;
}

void CudaStreamContext::SyncRecycleEvent(cudaEvent_t event) {
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

KernelObserver* CudaStreamContext::GetKernelObserver() { return kernel_observer_.get(); }

ep::Stream* CudaStreamContext::stream() { return &stream_; }

REGISTER_STREAM_CONTEXT_CREATOR_WITH_STREAM_ID(
    DeviceType::kGPU, ([](const StreamId& stream_id) -> StreamContext* {
      CHECK_EQ(stream_id.device_type(), DeviceType::kGPU);
      return new CudaStreamContext(stream_id.device_index());
    }));

}  // namespace oneflow

#endif
