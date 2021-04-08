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
#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

HashMap<std::string, InstrTypeId>* InstrTypeId4InstructionName() {
  static HashMap<std::string, InstrTypeId> map;
  return &map;
}

}  // namespace

void InstructionType::Compute(VirtualMachine* vm, Instruction* instruction) const {
  Compute(vm, instruction->mut_instr_msg());
}

void InstructionType::Infer(VirtualMachine* vm, Instruction* instruction) const {
  Infer(vm, instruction->mut_instr_msg());
}

const InstrTypeId& LookupInstrTypeId(const std::string& name) {
  const auto& map = *InstrTypeId4InstructionName();
  const auto& iter = map.find(name);
  CHECK(iter != map.end()) << "instruction type name: " << name;
  return iter->second;
}

void ForEachInstrTypeId(std::function<void(const InstrTypeId&)> DoEach) {
  for (const auto& pair : *InstrTypeId4InstructionName()) { DoEach(pair.second); }
}

HashMap<std::type_index, const InstructionType*>* InstructionType4TypeIndex() {
  static HashMap<std::type_index, const InstructionType*> map;
  return &map;
}

void RegisterInstrTypeId(const std::string& instruction_name, const StreamType* stream_type,
                         const InstructionType* instruction_type, InterpretType interpret_type) {
  InstrTypeId instr_type_id;
  instr_type_id.__Init__(stream_type, instruction_type, interpret_type);
  CHECK(InstrTypeId4InstructionName()->emplace(instruction_name, instr_type_id).second);
}

}  // namespace vm
}  // namespace oneflow
