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
#include "oneflow/core/vm/infer_stream_type.h"

namespace oneflow {
namespace vm {

class InstructionMsg;
class Instruction;
class VirtualMachine;

class InstructionType {
 public:
  virtual ~InstructionType() = default;

  bool IsSequential() const { return IsFrontSequential(); }
  virtual bool IsFrontSequential() const { return false; }
  virtual void Compute(Instruction* instruction) const = 0;
  virtual void Infer(Instruction* instruction) const = 0;

  virtual void Compute(VirtualMachine* vm, Instruction* instruction) const;
  virtual void Infer(VirtualMachine* vm, Instruction* instruction) const;
  virtual void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  virtual void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    LOG(FATAL) << "UNIMPLEMENTED";
  }

 protected:
  InstructionType() = default;
};

class InstrTypeId;
const InstrTypeId& LookupInstrTypeId(const std::string& instr_type_name);
void ForEachInstrTypeId(std::function<void(const InstrTypeId&)> DoEach);
void RegisterInstrTypeId(const std::string& instr_type_name, const StreamType* stream_type,
                         const InstructionType* instruction_type, InterpretType interpret_type);

HashMap<std::type_index, const InstructionType*>* InstructionType4TypeIndex();

template<typename T>
void TryRegisterInstructionType() {
  auto* map = InstructionType4TypeIndex();
  std::type_index type_index(typeid(T));
  if (map->find(type_index) == map->end()) { map->emplace(type_index, new T()); }
}

template<typename T>
const InstructionType* LookupInstructionType() {
  return InstructionType4TypeIndex()->at(typeid(T));
}

template<typename T>
void RegisterInstrTypeId(const std::string& instr_type_name, const StreamType* stream_type,
                         InterpretType interpret_type) {
  TryRegisterInstructionType<T>();
  RegisterInstrTypeId(instr_type_name, stream_type, LookupInstructionType<T>(), interpret_type);
}

template<typename T>
void RegisterInstructionType(const std::string& instr_type_name) {
  TryRegisterStreamType4TypeIndex<typename T::stream_type>();
  TryRegisterStreamType4TypeIndex<InferStreamType<typename T::stream_type>>();
  TryRegisterInferStreamTypeId<InferStreamType<typename T::stream_type>, typename T::stream_type>();
  RegisterInstrTypeId<T>(instr_type_name, LookupStreamType4TypeIndex<typename T::stream_type>(),
                         InterpretType::kCompute);
  RegisterInstrTypeId<T>(std::string("Infer-") + instr_type_name,
                         LookupStreamType4TypeIndex<InferStreamType<typename T::stream_type>>(),
                         InterpretType::kInfer);
}

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_TYPE_H_
