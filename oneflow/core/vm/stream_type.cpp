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
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/object_msg/object_msg.h"

namespace oneflow {
namespace vm {

namespace {

HashMap<StreamTypeId, StreamTypeId>* InferStreamTypeId4ComputeStreamTypeId() {
  static HashMap<StreamTypeId, StreamTypeId> map;
  return &map;
}

}  // namespace

HashMap<std::type_index, const StreamType*>* StreamType4TypeIndex() {
  static HashMap<std::type_index, const StreamType*> map;
  return &map;
}

const StreamTypeId& LookupInferStreamTypeId(const StreamTypeId& compute_stream_type_id) {
  return InferStreamTypeId4ComputeStreamTypeId()->at(compute_stream_type_id);
}

void StreamType::Run(Instruction* instruction) const {
  const auto& stream_type_id = instruction->stream().stream_id().stream_type_id();
  auto interpret_type = stream_type_id.interpret_type();
  if (interpret_type == InterpretType::kCompute) {
    Compute(instruction);
  } else if (interpret_type == InterpretType::kInfer) {
    Infer(instruction);
  } else {
    UNIMPLEMENTED();
  }
}

void StreamType::Run(VirtualMachine* vm, InstructionMsg* instr_msg) const {
  InterpretType interpret_type = instr_msg->instr_type_id().stream_type_id().interpret_type();
  if (interpret_type == InterpretType::kCompute) {
    Compute(vm, instr_msg);
  } else if (interpret_type == InterpretType::kInfer) {
    Infer(vm, instr_msg);
  } else {
    UNIMPLEMENTED();
  }
}

void StreamType::Run(VirtualMachine* vm, Instruction* instruction) const {
  auto interpret_type = instruction->stream().stream_id().stream_type_id().interpret_type();
  if (interpret_type == InterpretType::kCompute) {
    Compute(vm, instruction);
  } else if (interpret_type == InterpretType::kInfer) {
    Infer(vm, instruction);
  } else {
    UNIMPLEMENTED();
  }
}

void TryRegisterInferStreamTypeId(const StreamType* infer_stream_type,
                                  const StreamType* compute_stream_type) {
  StreamTypeId compute_stream_type_id;
  compute_stream_type_id.__Init__(compute_stream_type, InterpretType::kCompute);
  StreamTypeId infer_stream_type_id;
  infer_stream_type_id.__Init__(infer_stream_type, InterpretType::kInfer);
  InferStreamTypeId4ComputeStreamTypeId()->emplace(compute_stream_type_id, infer_stream_type_id);
}

}  // namespace vm
}  // namespace oneflow
