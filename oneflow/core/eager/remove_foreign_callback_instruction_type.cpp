#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/job/foreign_callback.h"

namespace oneflow {

namespace eager {

class RemoveForeignCallbackInstructionType : public vm::InstructionType {
 public:
  RemoveForeignCallbackInstructionType() = default;
  ~RemoveForeignCallbackInstructionType() override = default;

  using stream_type = vm::HostStreamType;

  void Infer(vm::Instruction* instruction) const override {
    // do nothing
  }

  void Compute(vm::Instruction* instruction) const override {
    FlatMsgView<RemoveForeignCallbackInstrOperand> args(instruction->instr_msg().operand());
    Global<ForeignCallback>::Get()->RemoveForeignCallback(args->unique_callback_id());
  }
};

COMMAND(vm::RegisterInstructionType<RemoveForeignCallbackInstructionType>("RemoveForeignCallback"));

}  // namespace eager

}  // namespace oneflow
