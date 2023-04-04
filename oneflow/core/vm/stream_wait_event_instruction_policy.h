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

#ifndef ONEFLOW_CORE_VM_STREAM_WAIT_EVENT_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_STREAM_WAIT_EVENT_INSTRUCTION_POLICY_H_

#include <functional>
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/common/op_args_reserved_size.h"
#include "oneflow/core/common/small_vector.h"
#include "oneflow/core/vm/stream_record_event_instruction_policy.h"

namespace oneflow {
namespace vm {

class Stream;

class StreamWaitEventInstructionPolicy final : public vm::InstructionPolicy {
 public:
  StreamWaitEventInstructionPolicy(
      const small_vector<intrusive::shared_ptr<LocalDepObject>>& dependences,
      const std::shared_ptr<StreamRecordEventInstructionPolicy>&
          stream_record_event_instruction_policy);
  ~StreamWaitEventInstructionPolicy() = default;

  std::string DebugName(const vm::Instruction&) const override { return "StreamWaitEvent"; }

  void DeleteInstructionStatus(Instruction* instruction) override;
  Maybe<void> Prepare(vm::Instruction* instruction) override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) override;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

 private:
  small_vector<intrusive::shared_ptr<LocalDepObject>> dependences_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
  std::shared_ptr<StreamRecordEventInstructionPolicy> stream_record_event_instruction_policy_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_WAIT_EVENT_INSTRUCTION_POLICY_H_
