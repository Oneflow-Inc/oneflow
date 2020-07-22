#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.msg.h"

namespace oneflow {
namespace eager {

class LazyReferenceInstructionType : public vm::InstructionType {
 public:
  LazyReferenceInstructionType() = default;
  virtual ~LazyReferenceInstructionType() override = default;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(LazyReferenceInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::MutOperand, eager_blob);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::SymbolOperand, lbn_sym_id);
  FLAT_MSG_VIEW_END(LazyReferenceInstruction);
  // clang-format on

  void Infer(vm::Instruction* instruction) const override { CHECK_OK(Run(instruction)); };
  void Compute(vm::Instruction* instruction) const override{
      // do nothing
  };

 private:
  Maybe<void> Run(vm::Instruction* instruction) const;
};

}  // namespace eager
}  // namespace oneflow
