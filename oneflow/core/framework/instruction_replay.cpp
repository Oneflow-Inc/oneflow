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

#include <list>
#include "oneflow/core/framework/instruction_replay.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/instruction.h"

namespace oneflow {

namespace {

bool* RecordingInstructionsFlag() {
  static thread_local bool recording_instruction = false;
  return &recording_instruction;
}

std::list<intrusive::shared_ptr<vm::InstructionMsg>>* RecordedInstructionList() {
  static thread_local std::list<intrusive::shared_ptr<vm::InstructionMsg>> list;
  return &list;
}

}  // namespace

namespace debug {

bool RecordingInstructions() { return *RecordingInstructionsFlag(); }

void StartRecordingInstructions() { *RecordingInstructionsFlag() = true; }

void EndRecordingInstructions() { *RecordingInstructionsFlag() = false; }

void ClearRecordedInstructions() { RecordedInstructionList()->clear(); }

void RecordInstruction(const intrusive::shared_ptr<vm::InstructionMsg>& instruction) {
  RecordedInstructionList()->push_back(instruction);
}

void ReplayInstructions() {
  vm::InstructionMsgList instr_msg_list;
  for (const auto& instr_msg : *RecordedInstructionList()) {
    instr_msg_list.EmplaceBack(instr_msg->Clone());
  }
  CHECK_JUST(vm::Run(&instr_msg_list));
}

}  // namespace debug

}  // namespace oneflow
