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
#ifndef ONEFLOW_CORE_VM_EP_STREAM_POLICY_BASE_H_
#define ONEFLOW_CORE_VM_EP_STREAM_POLICY_BASE_H_

#include "oneflow/core/vm/stream_policy.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/vm/ep_event.h"
#include "oneflow/core/vm/bin_allocator.h"
#include "oneflow/core/vm/thread_safe_guard.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {
namespace vm {

class EpStreamPolicyBase : public StreamPolicy {
 public:
  EpStreamPolicyBase(Symbol<Device> device, std::unique_ptr<vm::Allocator>&& backend_allocator)
      : device_(device),
        ep_event_provier_(),
        ep_stream_(nullptr),
        ep_allocator_(std::move(backend_allocator)) {}
  virtual ~EpStreamPolicyBase() override {
    if (ep_stream_ != nullptr) {
      CHECK(ep_device_);
      ep_device_->DestroyStream(ep_stream_);
    }
  }

  ep::Stream* stream() override { return GetOrCreateEpStream(); }

  vm::Allocator* mut_allocator() override { return ep_allocator_.get(); }

  DeviceType device_type() const override { return device_->enum_type(); }

  EpEventProvider* ep_event_provider() {
    if (unlikely(ep_event_provier_ == nullptr)) {
      ep_event_provier_.reset(new SingleThreadEpEventProvider(GetOrCreateEpDevice()));
    }
    return ep_event_provier_.get();
  }

  ep::Device* GetOrCreateEpDevice() const {
    if (unlikely(ep_device_ == nullptr)) {
      ep_device_ = Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(device_->enum_type(),
                                                                          device_->device_id());
      CHECK(ep_device_);
    }
    return ep_device_.get();
  }

  bool SupportingTransportInstructions() const override { return true; }

  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;

  bool QueryInstructionStatusLaunched(const Stream& stream,
                                      const InstructionStatusBuffer& status_buffer) const override;

  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;

  void Run(Instruction* instruction) const override;

 private:
  ep::Stream* GetOrCreateEpStream() const {
    if (unlikely(ep_stream_ == nullptr)) {
      ep_stream_ = GetOrCreateEpDevice()->CreateStream();
      CHECK(ep_stream_ != nullptr);
    }
    return ep_stream_;
  }

  Symbol<Device> device_;
  std::unique_ptr<EpEventProvider> ep_event_provier_;
  mutable std::shared_ptr<ep::Device> ep_device_;
  mutable ep::Stream* ep_stream_;
  std::unique_ptr<vm::Allocator> ep_allocator_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_EP_STREAM_POLICY_BASE_H_
