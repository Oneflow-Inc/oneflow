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
#include "oneflow/core/vm/stream_desc.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/intrusive/flat_msg_view.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class NewSymbolInstructionType final : public InstructionType {
 public:
  NewSymbolInstructionType() = default;
  ~NewSymbolInstructionType() override = default;

  bool ResettingIdToObjectMap() const override { return true; }

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(NewSymbolInstruction);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, symbol_id);
  FLAT_MSG_VIEW_END(NewSymbolInstruction);
  // clang-format on

  void Infer(VirtualMachineEngine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetTypeId>(vm, instr_msg);
  }
  void Compute(VirtualMachineEngine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetValueId>(vm, instr_msg);
  }
  void Infer(Instruction*) const override { UNIMPLEMENTED(); }
  void Compute(Instruction*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(VirtualMachineEngine* vm, InstructionMsg* instr_msg) const {
    FlatMsgView<NewSymbolInstruction> view;
    CHECK(view.Match(instr_msg->operand()));
    FOR_RANGE(int, i, 0, view->symbol_id_size()) {
      int64_t symbol_id = GetLogicalObjectId(view->symbol_id(i));
      auto logical_object = intrusive::make_shared<LogicalObject>(symbol_id);
      CHECK(vm->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
      auto* global_device_id2mirrored_object =
          logical_object->mut_global_device_id2mirrored_object();
      auto mirrored_object = intrusive::make_shared<MirroredObject>(logical_object.Mutable(), 0);
      CHECK(global_device_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
    }
  }
};
COMMAND(RegisterInstructionType<NewSymbolInstructionType>("NewSymbol"));

void ControlStreamType::Infer(VirtualMachineEngine* vm, InstructionMsg* instr_msg) const {
  const auto& instr_type_id = instr_msg->instr_type_id();
  CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kInfer);
  instr_type_id.instruction_type().Infer(vm, instr_msg);
}

void ControlStreamType::Infer(VirtualMachineEngine* vm, Instruction* instruction) const {
  const auto& instr_type_id = instruction->instr_msg().instr_type_id();
  CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kInfer);
  instr_type_id.instruction_type().Infer(vm, instruction);
  auto* status_buffer = instruction->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

void ControlStreamType::Compute(VirtualMachineEngine* vm, InstructionMsg* instr_msg) const {
  const auto& instr_type_id = instr_msg->instr_type_id();
  CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
  instr_type_id.instruction_type().Compute(vm, instr_msg);
}

void ControlStreamType::Compute(VirtualMachineEngine* vm, Instruction* instruction) const {
  const auto& instr_type_id = instruction->instr_msg().instr_type_id();
  CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
  instr_type_id.instruction_type().Compute(vm, instruction);
  auto* status_buffer = instruction->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

void ControlStreamType::InitInstructionStatus(const Stream& stream,
                                              InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void ControlStreamType::DeleteInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  auto* ptr = NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data());
  ptr->~NaiveInstrStatusQuerier();
}

bool ControlStreamType::QueryInstructionStatusLaunched(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void ControlStreamType::Compute(Instruction* instruction) const { UNIMPLEMENTED(); }

intrusive::shared_ptr<StreamDesc> ControlStreamType::MakeStreamDesc(const Resource& resource,
                                                                    int64_t this_machine_id) const {
  auto ret = intrusive::make_shared<StreamDesc>();
  ret->mut_stream_type_id()->__Init__(LookupStreamType4TypeIndex<ControlStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
