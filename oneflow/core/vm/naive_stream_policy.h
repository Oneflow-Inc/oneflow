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
#ifndef ONEFLOW_CORE_VM_NAIVE_STREAM_POLICY_H_
#define ONEFLOW_CORE_VM_NAIVE_STREAM_POLICY_H_

#include "oneflow/core/vm/stream_policy.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/lazy_job_device_context.h"

namespace oneflow {
namespace vm {

class NaiveStreamPolicy final : public StreamPolicy {
 public:
  NaiveStreamPolicy(const StreamType* stream_type, std::unique_ptr<DeviceCtx>&& device_ctx)
      : stream_type_(stream_type), device_ctx_(std::move(device_ctx)) {}

  ~NaiveStreamPolicy() override = default;

  ep::Stream* stream() override {
    if (device_ctx_) {
      return device_ctx_->stream();
    } else {
      return nullptr;
    }
  }
  vm::Allocator* mut_allocator() override {
    if (device_ctx_) {
      return device_ctx_->mut_allocator();
    } else {
      return nullptr;
    }
  }
  DeviceType device_type() const override {
    if (device_ctx_) {
      return device_ctx_->device_type();
    } else {
      return DeviceType::kInvalidDevice;
    }
  }

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override {
    stream_type_->InitInstructionStatus(stream, status_buffer);
  }
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override {
    stream_type_->DeleteInstructionStatus(stream, status_buffer);
  }
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override {
    return stream_type_->QueryInstructionStatusDone(stream, status_buffer);
  }
  void Run(Instruction* instruction) const override { stream_type_->Run(instruction); }

  bool OnSchedulerThread(StreamRole stream_role) const override {
    return stream_type_->OnSchedulerThread(stream_role);
  }

  bool SupportingTransportInstructions() const override {
    return stream_type_->SupportingTransportInstructions();
  }

  const std::unique_ptr<DeviceCtx>& device_ctx() const { return device_ctx_; }

 private:
  const StreamType* stream_type_;
  std::unique_ptr<DeviceCtx> device_ctx_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_NAIVE_STREAM_POLICY_H_
