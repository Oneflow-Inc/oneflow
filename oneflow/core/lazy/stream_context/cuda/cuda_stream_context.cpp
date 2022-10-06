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
#include "oneflow/core/lazy/stream_context/include/stream_context.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/cuda_check_numerics_kernel_observer.h"
#include "oneflow/core/graph/stream_id.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/cuda/cuda_device.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/common/channel.h"

#ifdef WITH_CUDA
#include <cublas_v2.h>

namespace oneflow {

namespace {

class CudaStreamContext : public StreamContext, public KernelObserverProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaStreamContext);
  explicit CudaStreamContext(int device_index);
  ~CudaStreamContext() override;

  Maybe<void> AddCallback(std::function<void()> callback) override;
  DeviceType device_type() const override { return DeviceType::kCUDA; }
  KernelObserver* GetKernelObserver() override;

  ep::Stream* stream() override;

 private:
  ep::CudaStream* stream_;
  Channel<std::pair<ep::Event*, std::function<void()>>> cb_event_chan_;
  std::thread poller_thread_;
  int device_index_;
  std::unique_ptr<KernelObserver> kernel_observer_;
  std::shared_ptr<ep::CudaDevice> device_;
};

CudaStreamContext::CudaStreamContext(int device_index)
    : stream_(nullptr), device_index_(device_index) {
  CudaCurrentDeviceGuard guard(device_index_);
  device_ = std::dynamic_pointer_cast<ep::CudaDevice>(
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCUDA, device_index));
  CHECK(device_);
  stream_ = dynamic_cast<ep::CudaStream*>(device_->CreateStream());
  CHECK(stream_ != nullptr);

  std::vector<std::shared_ptr<KernelObserver>> kernel_observers;
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS", false)) {
    LOG(WARNING) << "Environment variable ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS has been set "
                    "to a truthy "
                    "value, it will impact performance";
    kernel_observers.emplace_back(new CudaCheckNumericsKernelObserver());
  }
  kernel_observer_.reset(new ChainKernelObserver(kernel_observers));

  poller_thread_ = std::thread([this]() {
    CHECK_JUST(stream_->OnExecutionContextSetup());
    OF_PROFILER_NAME_THIS_HOST_THREAD("_cuda" + std::to_string(device_index_) + " Poller : ("
                                      + std::to_string(device_index_) + ")");
    std::pair<ep::Event*, std::function<void()>> cb_event;
    while (cb_event_chan_.Receive(&cb_event) == kChannelStatusSuccess) {
      CHECK_JUST(cb_event.first->Sync());
      cb_event.second();
      device_->DestroyEvent(cb_event.first);
    }
    CHECK_JUST(stream_->OnExecutionContextTeardown());
  });
}

CudaStreamContext::~CudaStreamContext() {
  CudaCurrentDeviceGuard guard(device_index_);
  cb_event_chan_.Close();
  poller_thread_.join();
  device_->DestroyStream(stream_);
}

Maybe<void> CudaStreamContext::AddCallback(std::function<void()> callback) {
  ep::Event* event = device_->CreateEvent();
  stream_->RecordEvent(event);
  cb_event_chan_.Send(std::make_pair(event, std::move(callback)));
  return Maybe<void>::Ok();
}

KernelObserver* CudaStreamContext::GetKernelObserver() { return kernel_observer_.get(); }

ep::Stream* CudaStreamContext::stream() { return stream_; }

REGISTER_STREAM_CONTEXT_CREATOR_WITH_STREAM_ID(
    DeviceType::kCUDA, ([](const StreamId& stream_id) -> StreamContext* {
      CHECK_EQ(stream_id.device_type(), DeviceType::kCUDA);
      return new CudaStreamContext(stream_id.device_index());
    }));

}  // namespace

}  // namespace oneflow

#endif
