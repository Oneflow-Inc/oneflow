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
#ifndef ONEFLOW_CORE_EAGER_BLOB_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_EAGER_BLOB_INSTRUCTION_TYPE_H_

#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/ep_event.h"
#include "oneflow/core/vm/ep_device_context.h"

namespace oneflow {
namespace vm {

class AccessBlobByCallbackInstructionType final : public vm::InstructionType {
 public:
  AccessBlobByCallbackInstructionType() = default;
  ~AccessBlobByCallbackInstructionType() override = default;

  std::string DebugName(const vm::Instruction& instruction) const override {
    return "AccessBlobByCallback";
  }
  Maybe<void> Prepare(vm::Instruction* instruction) const override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) const override;
};

class EpRecordEventInstructionType final : public vm::InstructionType {
 public:
  EpRecordEventInstructionType() = default;
  ~EpRecordEventInstructionType() override = default;

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAsTailOnly; }

  void InitInstructionStatus(Instruction* instruction) const override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_type().InitInstructionStatus(*stream, status_buffer);
    auto* ep_device_ctx = static_cast<EpDeviceCtx*>(stream->device_ctx().get());
    auto* ep_event_provider = ep_device_ctx->ep_event_provider();
    const auto& ep_event = CHECK_NOTNULL(ep_event_provider)->GetReusedEpEvent();
    auto* data_ptr = status_buffer->mut_buffer();
    EpOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_ep_event(ep_event);
  }
  Maybe<void> Prepare(vm::Instruction* instruction) const override { return Maybe<void>::Ok(); }
  std::string DebugName(const vm::Instruction&) const override { return "RecordEvent"; }
  void Compute(vm::Instruction* instruction) const override {}
};
}  // namespace vm

struct GetRecordEventInstructionType : public StreamRoleVisitor<GetRecordEventInstructionType> {
  static Maybe<const vm::InstructionType*> VisitCompute(DeviceType device_type) {
    return SingletonPtr<vm::EpRecordEventInstructionType>();
  }
  static Maybe<const vm::InstructionType*> VisitHost2Device(DeviceType device_type) {
    return SingletonPtr<vm::EpRecordEventInstructionType>();
  }
  static Maybe<const vm::InstructionType*> VisitDevice2Host(DeviceType device_type) {
    return SingletonPtr<vm::EpRecordEventInstructionType>();
  }
  static Maybe<const vm::InstructionType*> VisitSyncedLaunchedCommNet(DeviceType device_type) {
    return SingletonPtr<vm::EpRecordEventInstructionType>();
  }
  static Maybe<const vm::InstructionType*> VisitAsyncedLaunchedCommNet(DeviceType device_type) {
    return SingletonPtr<vm::EpRecordEventInstructionType>();
  }
  static Maybe<const vm::InstructionType*> VisitBarrier(DeviceType device_type) {
    UNIMPLEMENTED_THEN_RETURN();
  }
  static Maybe<const vm::InstructionType*> VisitCriticalSection(DeviceType device_type) {
    UNIMPLEMENTED_THEN_RETURN();
  }
  static Maybe<const vm::InstructionType*> VisitLazyJobLauncher(DeviceType device_type) {
    UNIMPLEMENTED_THEN_RETURN();
  }
  static Maybe<const vm::InstructionType*> VisitPinnedCompute(DeviceType device_type) {
    return VisitCompute(device_type);
  }
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_BLOB_INSTRUCTION_TYPE_H_
