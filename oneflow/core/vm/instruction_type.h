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
#ifndef ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_

#include <glog/logging.h>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

class Instruction;

enum InstructionFuseType {
  kInvalidInstructionFuseType = 0,
  kDisableInstructionFuse,
  kEnableInstructionFuseAtAnyPosition,
  kEnableInstructionFuseAsTailOnly,
};

class InstructionType {
 public:
  virtual ~InstructionType() = default;

  Maybe<void> PrepareIf(Instruction* instruction) const {
    OF_PROFILER_RANGE_GUARD(std::string("Prepare:") + DebugName(*instruction));
    InitOrCheckInputBlobsMemPtrForAllocationCompuationPipelining(instruction);
    return Prepare(instruction);
  }

  void ComputeIf(Instruction* instruction) const {
    OF_PROFILER_RANGE_GUARD(std::string("Compute:") + DebugName(*instruction));
    Compute(instruction);
  }

  virtual bool IsBarrier() const { return false; }
  virtual InstructionFuseType fuse_type() const { return kDisableInstructionFuse; }
  void InitInstructionStatusIf(Instruction* instruction) const {
    InitInstructionStatus(instruction);
  }

  void DeleteInstructionStatusIf(Instruction* instruction) const {
    DeleteInstructionStatus(instruction);
  }

  virtual std::string DebugName(const Instruction&) const = 0;

 protected:
  InstructionType() = default;

 private:
  // Allocating tensors, deallocating tensors, preparing opkernel states and preparing opkernel
  // caches.
  virtual Maybe<void> Prepare(Instruction* instruction) const = 0;

  virtual void Compute(Instruction* instruction) const = 0;

  virtual void InitInstructionStatus(Instruction* instruction) const;
  virtual void DeleteInstructionStatus(Instruction* instruction) const;
  void InitOrCheckInputBlobsMemPtrForAllocationCompuationPipelining(Instruction* instruction) const;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
