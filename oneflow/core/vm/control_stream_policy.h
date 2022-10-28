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
#ifndef ONEFLOW_CORE_VM_CONTROL_STREAM_POLICY_H_
#define ONEFLOW_CORE_VM_CONTROL_STREAM_POLICY_H_

#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/stream_policy.h"
#include "oneflow/core/vm/vm_object.h"

namespace oneflow {
namespace vm {

class ControlStreamPolicy final : public StreamPolicy {
 public:
  ControlStreamPolicy() = default;
  ~ControlStreamPolicy() = default;

  vm::Allocator* mut_allocator() override { return (vm::Allocator*)nullptr; }

  DeviceType device_type() const override {
    PRINT_BUG_PROMPT_AND_ABORT();
    return DeviceType::kInvalidDevice;
  }

  ep::Stream* stream() override {
    PRINT_BUG_PROMPT_AND_ABORT();
    return nullptr;
  }

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override {
    static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
    NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer());
  }
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override {
    auto* ptr = NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer());
    ptr->~NaiveInstrStatusQuerier();
  }
  bool QueryInstructionStatusLaunched(const Stream& stream,
                                      const InstructionStatusBuffer& status_buffer) const override {
    return NaiveInstrStatusQuerier::Cast(status_buffer.buffer())->launched();
  }
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override {
    return NaiveInstrStatusQuerier::Cast(status_buffer.buffer())->done();
  }
  void Run(Instruction* instruction) const override {
    instruction->Compute();
    auto* status_buffer = instruction->mut_status_buffer();
    NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer())->set_done();
  }

  bool OnSchedulerThread(StreamType) const override { return true; }
  bool SupportingTransportInstructions() const override { return false; }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONTROL_STREAM_POLICY_H_
