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

#ifndef ONEFLOW_CORE_VM_STREAM_WAIT_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_STREAM_WAIT_INSTRUCTION_TYPE_H_

#include <functional>
#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/vm/instruction_type.h"

namespace oneflow {
class EpEvent;
namespace vm {

class Stream;

class StreamWaitPhyInstrOperand : public PhyInstrOperand {
 public:
  StreamWaitPhyInstrOperand(std::vector<intrusive::shared_ptr<LocalDepObject>>&& dependences,
                            vm::Stream* from_vm_stream)
      : dependences_(std::move(dependences)),
        input_dependences_(),
        output_dependences_(),
        from_vm_stream_(from_vm_stream) {
    const auto& Insert = SetInserter(&output_dependences_);
    for (const auto& dep : dependences) { Insert(dep.get()); }
    stream_sequential_dependence_ = nullptr;
  }

  ~StreamWaitPhyInstrOperand() = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {}

  vm::Stream* mut_from_vm_stream() { return from_vm_stream_; }

  std::shared_ptr<EpEvent>& mut_ep_event() { return ep_event_; }

 private:
  std::vector<intrusive::shared_ptr<LocalDepObject>> dependences_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
  vm::Stream* from_vm_stream_;
  std::shared_ptr<EpEvent> ep_event_;
};

class StreamWaitInstructionType final : public vm::InstructionType {
 public:
  StreamWaitInstructionType() = default;

  std::string DebugName(const vm::Instruction&) const override { return "StreamWait"; }

  bool Prescheduleable(const Stream* src, const Stream* dst) const override;
  void InitInstructionStatus(Instruction* instruction) const override;
  void DeleteInstructionStatus(Instruction* instruction) const override;
  Maybe<void> Prepare(vm::Instruction* instruction) const override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) const override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_WAIT_INSTRUCTION_TYPE_H_
