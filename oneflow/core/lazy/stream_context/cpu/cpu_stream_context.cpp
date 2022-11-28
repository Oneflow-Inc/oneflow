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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/cpu_check_numerics_kernel_observer.h"
#include "oneflow/core/graph/stream_id.h"
#include "oneflow/core/ep/cpu/cpu_stream.h"
#include "oneflow/core/ep/cpu/cpu_device.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

class CpuStreamContext : public StreamContext, public KernelObserverProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStreamContext);
  CpuStreamContext();
  ~CpuStreamContext() override;

  ep::Stream* stream() override;
  Maybe<void> AddCallback(std::function<void()> callback) override;
  KernelObserver* GetKernelObserver() override;
  DeviceType device_type() const override { return DeviceType::kCPU; }

 private:
  std::shared_ptr<ep::Device> device_;
  ep::Stream* stream_;
  std::unique_ptr<KernelObserver> kernel_observer_;
};

CpuStreamContext::CpuStreamContext() : stream_(nullptr) {
  device_ = Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(DeviceType::kCPU, 0);
  stream_ = device_->CreateStream();  // NOLINT
  std::vector<std::shared_ptr<KernelObserver>> kernel_observers;
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS", false)) {
    kernel_observers.emplace_back(new CpuCheckNumericsKernelObserver());
  }
  kernel_observer_.reset(new ChainKernelObserver(kernel_observers));
}

CpuStreamContext::~CpuStreamContext() { device_->DestroyStream(stream_); }

ep::Stream* CpuStreamContext::stream() { return stream_; }

Maybe<void> CpuStreamContext::AddCallback(std::function<void()> callback) {
  callback();
  return Maybe<void>::Ok();
}

KernelObserver* CpuStreamContext::GetKernelObserver() { return kernel_observer_.get(); }

REGISTER_STREAM_CONTEXT_CREATOR_WITH_STREAM_ID(DeviceType::kCPU,
                                               ([](const StreamId& stream_id) -> StreamContext* {
                                                 return new CpuStreamContext();
                                               }));

}  // namespace oneflow
