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
#include "oneflow/core/vm/stream_type.h"

namespace oneflow {
namespace vm {

class InstructionMsg;
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

  bool IsSequential() const { return IsFrontSequential(); }
  virtual bool IsFrontSequential() const { return false; }
  virtual InstructionFuseType fuse_type() const { return kDisableInstructionFuse; }
  virtual void Compute(Instruction* instruction) const = 0;

  virtual void ComputeInFuseMode(InstructionMsg* instr_msg) const { LOG(FATAL) << "UNIMPLEMENTED"; }
  void InitInstructionStatusIf(Instruction* instruction) const {
    InitInstructionStatus(instruction);
  }
  void DeleteInstructionStatusIf(Instruction* instruction) const {
    DeleteInstructionStatus(instruction);
  }

  virtual std::string DebugOpTypeName(const InstructionMsg&) const { return ""; }

 protected:
  InstructionType() = default;

 private:
  virtual void InitInstructionStatus(Instruction* instruction) const;
  virtual void DeleteInstructionStatus(Instruction* instruction) const;
};

class InstrTypeId;
const InstrTypeId& LookupInstrTypeId(const std::string& instr_type_name);
void ForEachInstrTypeId(std::function<void(const InstrTypeId&)> DoEach);
void RegisterInstrTypeId(const std::string& instr_type_name, const StreamType* stream_type,
                         const InstructionType* instruction_type);

template<typename T>
const InstructionType* StaticGlobalInstructionType() {
  static const InstructionType* instruction_type = new T();
  return instruction_type;
}

template<typename T>
void RegisterInstrTypeId(const std::string& instr_type_name, const StreamType* stream_type) {
  RegisterInstrTypeId(instr_type_name, stream_type, StaticGlobalInstructionType<T>());
}

template<typename T>
void RegisterInstructionType(const std::string& instr_type_name) {
  RegisterInstrTypeId<T>(instr_type_name, StaticGlobalStreamType<typename T::stream_type>());
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
