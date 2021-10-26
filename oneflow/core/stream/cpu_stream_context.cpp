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
#include "oneflow/core/stream/stream_context.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/cpu_check_numerics_kernel_observer.h"

namespace oneflow {

class CpuStreamContext : public StreamContext, public KernelObserverProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStreamContext);
  explicit CpuStreamContext();
  virtual ~CpuStreamContext();

  Maybe<void> AddCallback(std::function<void()> callback) override;
  Maybe<void> Sync() override;
  KernelObserver* GetKernelObserver() override;
  DeviceType device_type() const override { return DeviceType::kCPU; }

 private:
  std::unique_ptr<KernelObserver> kernel_observer_;
};

CpuStreamContext::CpuStreamContext() {
  std::vector<std::shared_ptr<KernelObserver>> kernel_observers;
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS", false)) {
    kernel_observers.emplace_back(new CpuCheckNumericsKernelObserver());
  }
  kernel_observer_.reset(new ChainKernelObserver(kernel_observers));
}

CpuStreamContext::~CpuStreamContext() = default;

Maybe<void> CpuStreamContext::AddCallback(std::function<void()> callback) {
  callback();
  return Maybe<void>::Ok();
}

Maybe<void> CpuStreamContext::Sync() { return Maybe<void>::Ok(); }

KernelObserver* CpuStreamContext::GetKernelObserver() { return kernel_observer_.get(); }

REGISTER_STREAM_CONTEXT_CREATOR_WITH_STREAM_ID(DeviceType::kCPU,
                                               ([](const StreamId& stream_id) -> StreamContext* {
                                                 return new CpuStreamContext();
                                               }));

}  // namespace oneflow
