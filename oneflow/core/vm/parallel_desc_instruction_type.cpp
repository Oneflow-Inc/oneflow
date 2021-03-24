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
#include "oneflow/core/common/util.h"
#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {
namespace vm {

class NewParallelDescSymbolInstructionType final : public InstructionType {
 public:
  NewParallelDescSymbolInstructionType() = default;
  ~NewParallelDescSymbolInstructionType() override = default;

  using stream_type = ControlStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(ParallelDescObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(int64_t, logical_object_id);
  FLAT_MSG_VIEW_END(ParallelDescObjectInstrOperand);
  // clang-format on

  void Infer(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetTypeId>(vm, instr_msg);
  }
  void Compute(VirtualMachine* vm, InstructionMsg* instr_msg) const override {
    Run<&IdUtil::GetValueId>(vm, instr_msg);
  }
  void Infer(Instruction*) const override { UNIMPLEMENTED(); }
  void Compute(Instruction*) const override { UNIMPLEMENTED(); }

 private:
  template<int64_t (*GetLogicalObjectId)(int64_t)>
  void Run(VirtualMachine* vm, InstructionMsg* instr_msg) const {
    FlatMsgView<ParallelDescObjectInstrOperand> view(instr_msg->operand());
    FOR_RANGE(int, i, 0, view->logical_object_id_size()) {
      int64_t logical_object_id = GetLogicalObjectId(view->logical_object_id(i));
      auto logical_object = ObjectMsgPtr<LogicalObject>::NewFrom(vm->mut_vm_thread_only_allocator(),
                                                                 logical_object_id);
      CHECK(vm->mut_id2logical_object()->Insert(logical_object.Mutable()).second);
      auto* global_device_id2mirrored_object =
          logical_object->mut_global_device_id2mirrored_object();
      auto mirrored_object =
          ObjectMsgPtr<MirroredObject>::NewFrom(vm->mut_allocator(), logical_object.Mutable(), 0);
      {
        const auto& parallel_desc =
            Global<symbol::Storage<ParallelDesc>>::Get()->GetPtr(view->logical_object_id(i));
        auto* rw_mutexed_object = mirrored_object->mut_rw_mutexed_object();
        rw_mutexed_object->Init<ObjectWrapper<ParallelDesc>>(parallel_desc);
      }
      CHECK(global_device_id2mirrored_object->Insert(mirrored_object.Mutable()).second);
    }
  }
};
COMMAND(Global<symbol::Storage<ParallelDesc>>::SetAllocated(new symbol::Storage<ParallelDesc>()));
COMMAND(RegisterInstructionType<NewParallelDescSymbolInstructionType>("NewParallelDescSymbol"));

}  // namespace vm
}  // namespace oneflow
