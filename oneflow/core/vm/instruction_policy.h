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
#ifndef ONEFLOW_CORE_VM_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_POLICY_H_

#include <functional>
#include <vector>
#include <memory>
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/vm/instruction_fuse_type.h"
#include "oneflow/core/vm/vm_object.h"

namespace oneflow {
namespace vm {

class EagerBlobObject;
class Stream;

class InstructionPolicy {
 public:
  virtual ~InstructionPolicy() = default;

  // Same stream.
  virtual bool Prescheduleable(const vm::Stream* src, const vm::Stream* dst) const {
    return src == dst;
  }

  virtual const DependenceVector& input_dependences() const = 0;
  virtual const DependenceVector& output_dependences() const = 0;
  virtual Dependence* stream_sequential_dependence() const { return stream_sequential_dependence_; }

  virtual bool IsBarrier() const { return false; }
  virtual InstructionFuseType fuse_type() const { return kDisableInstructionFuse; }
  virtual std::string DebugName(const Instruction&) const = 0;

  Maybe<void> PrepareIf(Instruction* instruction) {
    OF_PROFILER_RANGE_GUARD(std::string("Prepare:") + DebugName(*instruction));
    return Prepare(instruction);
  }

  void ComputeIf(Instruction* instruction) {
    OF_PROFILER_RANGE_GUARD(std::string("Compute:") + DebugName(*instruction));
    Compute(instruction);
  }

  void InitInstructionStatusIf(Instruction* instruction) { InitInstructionStatus(instruction); }

  void DeleteInstructionStatusIf(Instruction* instruction) { DeleteInstructionStatus(instruction); }

 protected:
  InstructionPolicy() : stream_sequential_dependence_(nullptr) {}

  Dependence* stream_sequential_dependence_;

 private:
  // Usually for Allocating and deallocating tensors.
  virtual Maybe<void> Prepare(Instruction* instruction) = 0;
  virtual void Compute(Instruction* instruction) = 0;
  virtual void InitInstructionStatus(Instruction* instruction);
  virtual void DeleteInstructionStatus(Instruction* instruction);
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_POLICY_H_
