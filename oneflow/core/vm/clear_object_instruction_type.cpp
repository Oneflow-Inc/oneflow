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
#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/vm/object_wrapper.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/object.h"

namespace oneflow {
namespace vm {

class TryClearObjectInstructionType final : public InstructionType {
 public:
  TryClearObjectInstructionType() = default;
  ~TryClearObjectInstructionType() override = default;

  using stream_type = DeviceHelperStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(TryClearObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(MutOperand, object);
  FLAT_MSG_VIEW_END(TryClearObjectInstrOperand);
  // clang-format on

  void Infer(Instruction* instruction) const override {
    ForEachObject(instruction, [&](const MutOperand& operand) {
      TryClearObject(instruction->mut_type_mirrored_object(operand));
    });
  }
  void Compute(Instruction* instruction) const override {
    ForEachObject(instruction, [&](const MutOperand& operand) {
      TryClearObject(instruction->mut_type_mirrored_object(operand));
      TryClearObject(instruction->mut_value_mirrored_object(operand));
    });
  }

 private:
  template<typename DoEachT>
  void ForEachObject(Instruction* instruction, const DoEachT& DoEach) const {
    FlatMsgView<TryClearObjectInstrOperand> view;
    CHECK(view.Match(instruction->instr_msg().operand()));
    for (int i = 0; i < view->object_size(); ++i) { DoEach(view->object(i)); }
  }

  void TryClearObject(MirroredObject* mirrored_object) const {
    if (mirrored_object == nullptr) { return; }
    if (mirrored_object->rw_mutexed_object().ref_cnt() > 1) { return; }
    mirrored_object->mut_rw_mutexed_object()->reset_object();
  }
};
COMMAND(RegisterInstructionType<TryClearObjectInstructionType>("TryClearObject"));

}  // namespace vm
}  // namespace oneflow
