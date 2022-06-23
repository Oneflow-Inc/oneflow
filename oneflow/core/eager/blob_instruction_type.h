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

#include "oneflow/core/intrusive/flat_msg_view.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/vm/cuda_optional_event_record_status_querier.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/device/cuda_event.h"

namespace oneflow {
namespace vm {

class AccessBlobByCallbackInstructionType final : public vm::InstructionType {
 public:
  AccessBlobByCallbackInstructionType() = default;
  ~AccessBlobByCallbackInstructionType() override = default;

  std::string DebugName(const vm::InstructionMsg& instr_msg) const override {
    return "AccessBlobByCallback";
  }
  void Compute(vm::Instruction* instruction) const override;
  void ComputeInFuseMode(vm::InstructionMsg* instruction_msg) const override;

 private:
  void ComputeInstrMsg(const vm::InstructionMsg& instruction_msg) const;
};

class CpuRecordEventInstructionType final : public vm::InstructionType {
 public:
  CpuRecordEventInstructionType() = default;
  ~CpuRecordEventInstructionType() override = default;

  std::string DebugName(const vm::InstructionMsg& instr_msg) const override {
    return "RecordEvent";
  }
  void Compute(vm::Instruction* instruction) const override {}
};

#ifdef WITH_CUDA

class CudaRecordEventInstructionType final : public vm::InstructionType {
 public:
  CudaRecordEventInstructionType() = default;
  ~CudaRecordEventInstructionType() override = default;

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAsTailOnly; }

  void InitInstructionStatus(Instruction* instruction) const override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_type().InitInstructionStatus(*stream, status_buffer);
    auto* event_provider = dynamic_cast<QueryCudaEventProvider*>(stream->device_ctx().get());
    const auto& cuda_event = CHECK_NOTNULL(event_provider)->GetCudaEvent();
    auto* data_ptr = status_buffer->mut_buffer()->mut_data();
    CudaOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_cuda_event(cuda_event);
  }
  std::string DebugName(const vm::InstructionMsg& instr_msg) const override {
    return "RecordEvent";
  }
  void Compute(vm::Instruction* instruction) const override {}
};

#endif

}  // namespace vm

struct GetRecordEventInstructionType : public StreamRoleVisitor<GetRecordEventInstructionType> {
  static Maybe<const vm::InstructionType*> VisitCompute(DeviceType device_type) {
    return GetInstructionType(device_type);
  }
  static Maybe<const vm::InstructionType*> VisitHost2Device(DeviceType device_type) {
    return GetInstructionType(device_type);
  }
  static Maybe<const vm::InstructionType*> VisitDevice2Host(DeviceType device_type) {
    return GetInstructionType(device_type);
  }
  static Maybe<const vm::InstructionType*> VisitSyncedLaunchedCommNet(DeviceType device_type) {
    return GetInstructionType(device_type);
  }
  static Maybe<const vm::InstructionType*> VisitAsyncedLaunchedCommNet(DeviceType device_type) {
    return GetInstructionType(device_type);
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

 private:
  static Maybe<const vm::InstructionType*> GetInstructionType(DeviceType device_type) {
    if (device_type == DeviceType::kCPU) {
      return SingletonPtr<vm::CpuRecordEventInstructionType>();
    } else if (device_type == DeviceType::kCUDA) {
#ifdef WITH_CUDA
      return SingletonPtr<vm::CudaRecordEventInstructionType>();
#else
      UNIMPLEMENTED_THEN_RETURN();
#endif
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_BLOB_INSTRUCTION_TYPE_H_
