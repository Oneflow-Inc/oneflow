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
#ifndef ONEFLOW_CORE_EAGER_RELEASE_TENSOR_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_EAGER_RELEASE_TENSOR_INSTRUCTION_TYPE_H_

#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/eager/release_tensor_arg_phy_instr_operand.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/singleton_ptr.h"

namespace oneflow {

namespace vm {

class ReleaseTensorInstructionType : public vm::InstructionType {
 public:
  ReleaseTensorInstructionType() = default;
  virtual ~ReleaseTensorInstructionType() = default;

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAtAnyPosition; }

  void InitInstructionStatus(Instruction* instruction) const override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_policy().InitInstructionStatus(*stream, status_buffer);
    auto* data_ptr = status_buffer->mut_buffer();
    EpOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_ep_event(nullptr);
  }

 protected:
  const std::shared_ptr<vm::EagerBlobObject>& GetEagerBlobObject(
      const vm::Instruction& instruction) const {
    const auto& phy_instr_operand = instruction.phy_instr_operand();
    CHECK(static_cast<bool>(phy_instr_operand));
    const auto* ptr =
        dynamic_cast<const vm::ReleaseTensorArgPhyInstrOperand*>(phy_instr_operand.get());
    CHECK_NOTNULL(ptr);
    return ptr->eager_blob_object();
  }
  void Release(const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) const {
    CHECK_JUST(eager_blob_object->DeallocateBlobDataPtr());
  }
};

class FastReleaseTensorInstructionType final : public ReleaseTensorInstructionType {
 public:
  FastReleaseTensorInstructionType() = default;
  ~FastReleaseTensorInstructionType() override = default;

  std::string DebugName(const vm::Instruction& instruction) const override {
    return "ReleasePodTensor";
  }

  Maybe<void> Prepare(vm::Instruction* instruction) const override {
    const auto& eager_blob_object = GetEagerBlobObject(*instruction);
    DataType data_type = eager_blob_object->data_type();
    CHECK(IsPODDataType(data_type));
    Release(eager_blob_object);
    return Maybe<void>::Ok();
  }

  void Compute(vm::Instruction* instruction) const override {}
};

class SlowReleaseTensorInstructionType final : public ReleaseTensorInstructionType {
 public:
  SlowReleaseTensorInstructionType() = default;
  ~SlowReleaseTensorInstructionType() override = default;

  std::string DebugName(const vm::Instruction& instruction) const override {
    return "ReleaseNonPodTensor";
  }

  Maybe<void> Prepare(vm::Instruction* instruction) const override { return Maybe<void>::Ok(); }

  void Compute(vm::Instruction* instruction) const override {
    const auto& eager_blob_object = GetEagerBlobObject(*instruction);
    DataType data_type = eager_blob_object->data_type();
    CHECK(!IsPODDataType(data_type));
    Release(eager_blob_object);
  }
};

}  // namespace vm

struct GetReleaseInstructionType : public StreamRoleVisitor<GetReleaseInstructionType> {
  static Maybe<const vm::InstructionType*> VisitCompute(DataType data_type) {
    return GetReleaseTensorInstructionType(data_type);
  }
  static Maybe<const vm::InstructionType*> VisitHost2Device(DataType data_type) {
    return GetReleaseTensorInstructionType(data_type);
  }
  static Maybe<const vm::InstructionType*> VisitDevice2Host(DataType data_type) {
    return GetReleaseTensorInstructionType(data_type);
  }
  static Maybe<const vm::InstructionType*> VisitSyncedLaunchedCommNet(DataType data_type) {
    return GetReleaseTensorInstructionType(data_type);
  }
  static Maybe<const vm::InstructionType*> VisitAsyncedLaunchedCommNet(DataType data_type) {
    return GetReleaseTensorInstructionType(data_type);
  }
  static Maybe<const vm::InstructionType*> VisitBarrier(DataType data_type) {
    UNIMPLEMENTED_THEN_RETURN();
  }
  static Maybe<const vm::InstructionType*> VisitCriticalSection(DataType data_type) {
    UNIMPLEMENTED_THEN_RETURN();
  }
  static Maybe<const vm::InstructionType*> VisitLazyJobLauncher(DataType data_type) {
    UNIMPLEMENTED_THEN_RETURN();
  }
  static Maybe<const vm::InstructionType*> VisitPinnedCompute(DataType data_type) {
    return VisitCompute(data_type);
  }

 private:
  static Maybe<const vm::InstructionType*> GetReleaseTensorInstructionType(DataType data_type) {
    if (IsPODDataType(data_type)) {
      return SingletonPtr<vm::FastReleaseTensorInstructionType>();
    } else {
      return SingletonPtr<vm::SlowReleaseTensorInstructionType>();
    }
  }
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_RELEASE_TENSOR_INSTRUCTION_TYPE_H_
