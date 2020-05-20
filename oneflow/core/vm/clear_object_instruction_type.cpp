#include "oneflow/core/common/flat_msg_view.h"
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

class ClearObjectInstructionType final : public InstructionType {
 public:
  ClearObjectInstructionType() = default;
  ~ClearObjectInstructionType() override = default;

  using stream_type = DeviceHelperStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(ClearObjectInstrOperand);
    FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(MutOperand, object);
  FLAT_MSG_VIEW_END(ClearObjectInstrOperand);
  // clang-format on

  void Infer(Instruction* instruction) const override {
    ForEachObject(instruction, [&](const MutOperand& operand) {
      ClearObject(instruction->mut_type_mirrored_object(operand));
    });
  }
  void Compute(Instruction* instruction) const override {
    ForEachObject(instruction, [&](const MutOperand& operand) {
      ClearObject(instruction->mut_type_mirrored_object(operand));
      ClearObject(instruction->mut_value_mirrored_object(operand));
    });
  }

 private:
  template<typename DoEachT>
  void ForEachObject(Instruction* instruction, const DoEachT& DoEach) const {
    FlatMsgView<ClearObjectInstrOperand> view;
    CHECK(view.Match(instruction->instr_msg().operand()));
    for (int i = 0; i < view->object_size(); ++i) { DoEach(view->object(i)); }
  }

  void ClearObject(MirroredObject* mirrored_object) const {
    if (mirrored_object == nullptr) { return; }
    mirrored_object->clear_rw_mutexed_object();
  }
};
COMMAND(RegisterInstructionType<ClearObjectInstructionType>("ClearObject"));
COMMAND(RegisterLocalInstructionType<ClearObjectInstructionType>("LocalClearObject"));

}  // namespace vm
}  // namespace oneflow
