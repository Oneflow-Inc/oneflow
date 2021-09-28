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
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/vm/cpu_allocator.h"
#include "oneflow/core/kernel/chain_kernel_observer.h"
#include "oneflow/core/kernel/cpu_check_numerics_kernel_observer.h"
#include "oneflow/core/stream/execution_context_hook.h"

namespace oneflow {

class CpuStreamContext : public StreamContext,
                         public ExecutionContextHook,
                         public KernelObserverProvider,
                         public DeviceCtxProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStreamContext);
  explicit CpuStreamContext();
  virtual ~CpuStreamContext();

  Maybe<void> AddCallback(std::function<void()> callback) override;
  Maybe<void> Sync() override;
  std::shared_ptr<DeviceCtx> GetDeviceCtx() override;
  KernelObserver* GetKernelObserver() override;
  DeviceType device_type() const override { return DeviceType::kCPU; }

  Maybe<void> OnExecutionContextSetup() override;
  Maybe<void> OnExecutionContextTeardown() override;

 private:
  std::shared_ptr<DeviceCtx> device_ctx_;
  std::unique_ptr<KernelObserver> kernel_observer_;
};

namespace {

class DeviceCtxImpl final : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCtxImpl);
  explicit DeviceCtxImpl(CpuStreamContext* stream_ctx) : stream_ctx_(stream_ctx) {}
  ~DeviceCtxImpl() = default;

  std::unique_ptr<DeviceCtx> Copy() const {
    return std::unique_ptr<DeviceCtx>(new DeviceCtxImpl(stream_ctx_));
  }

  void SyncDevice() override {}
  void AddCallBack(std::function<void()> callback) const override { callback(); }

  vm::Allocator* mut_allocator() override { return Global<vm::CpuAllocator>::Get(); }

  DeviceType device_type() const override { return stream_ctx_->device_type(); }

  std::shared_ptr<EventRecord> MakeEventRecord() override {
    return std::make_shared<NaiveEventRecord>();
  }

 private:
  CpuStreamContext* stream_ctx_;
};

}  // namespace

Maybe<void> CpuStreamContext::OnExecutionContextSetup() {
  OF_PROFILER_NAME_THIS_HOST_THREAD("__CPU Actor Thread");
  return Maybe<void>::Ok();
}

Maybe<void> CpuStreamContext::OnExecutionContextTeardown() { return Maybe<void>::Ok(); }

CpuStreamContext::CpuStreamContext() {
  std::vector<std::shared_ptr<KernelObserver>> kernel_observers;
  if (ParseBooleanFromEnv("ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS", false)) {
    kernel_observers.emplace_back(new CpuCheckNumericsKernelObserver());
  }
  kernel_observer_.reset(new ChainKernelObserver(kernel_observers));
  device_ctx_.reset(new DeviceCtxImpl(this));
}

CpuStreamContext::~CpuStreamContext() = default;

Maybe<void> CpuStreamContext::AddCallback(std::function<void()> callback) {
  callback();
  return Maybe<void>::Ok();
}

Maybe<void> CpuStreamContext::Sync() { return Maybe<void>::Ok(); }

std::shared_ptr<DeviceCtx> CpuStreamContext::GetDeviceCtx() { return device_ctx_; }

KernelObserver* CpuStreamContext::GetKernelObserver() { return kernel_observer_.get(); }

REGISTER_STREAM_CONTEXT_CREATOR_WITH_STREAM_ID(DeviceType::kCPU,
                                               ([](const StreamId& stream_id) -> StreamContext* {
                                                 return new CpuStreamContext();
                                               }));

}  // namespace oneflow
