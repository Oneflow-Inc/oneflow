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
#include "oneflow/core/eager/release_tensor_arg_phy_instr_operand.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/cuda_optional_event_record_status_querier.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/singleton_ptr.h"

namespace oneflow {

namespace vm {

class ReleaseTensorInstructionType : public vm::InstructionType {
 public:
  ReleaseTensorInstructionType() = default;
  ~ReleaseTensorInstructionType() override = default;

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAtAnyPosition; }

  void Release(const vm::InstructionMsg& instr_msg) const {
    const auto& phy_instr_operand = instr_msg.phy_instr_operand();
    CHECK(static_cast<bool>(phy_instr_operand));
    const auto* ptr =
        dynamic_cast<const vm::ReleaseTensorArgPhyInstrOperand*>(phy_instr_operand.get());
    CHECK_NOTNULL(ptr);
    CHECK_JUST(ptr->eager_blob_object()->DeallocateBlobDataPtr());
  }
  std::string DebugName(const vm::InstructionMsg& instr_msg) const override {
    return "ReleaseTensor";
  }
  void Compute(vm::Instruction* instruction) const override { Release(instruction->instr_msg()); }
  void ComputeInFuseMode(vm::InstructionMsg* instr_msg) const override { Release(*instr_msg); }
};

#ifdef WITH_CUDA

class CudaReleaseTensorInstructionType : public ReleaseTensorInstructionType {
 public:
  CudaReleaseTensorInstructionType() = default;
  ~CudaReleaseTensorInstructionType() override = default;

  void InitInstructionStatus(Instruction* instruction) const override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_type().InitInstructionStatus(*stream, status_buffer);
    auto* data_ptr = status_buffer->mut_buffer()->mut_data();
    CudaOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_cuda_event(nullptr);
  }
};

#endif

}  // namespace vm

struct GetReleaseInstructionType : public StreamRoleVisitor<GetReleaseInstructionType> {
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
      return SingletonPtr<vm::ReleaseTensorInstructionType>();
    } else if (device_type == DeviceType::kCUDA) {
#ifdef WITH_CUDA
      return SingletonPtr<vm::CudaReleaseTensorInstructionType>();
#else
      UNIMPLEMENTED_THEN_RETURN();
#endif
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_RELEASE_TENSOR_INSTRUCTION_TYPE_H_
