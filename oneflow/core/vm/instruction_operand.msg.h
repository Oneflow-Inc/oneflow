#ifndef ONEFLOW_CORE_VM_VM_INSTRUCTION_OPERAND_H_
#define ONEFLOW_CORE_VM_VM_INSTRUCTION_OPERAND_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"
#include "oneflow/core/vm/mirrored_object_id.msg.h"

namespace oneflow {
namespace vm {

enum OperandModifier {
  kConstModifier = 0,
  kDataMutableModifier,
  kTypeAndDataMutableModifier,
};

// clang-format off
template<OperandModifier modifier>
FLAT_MSG_BEGIN(LeveledMutableMirroredObjectOperand);
  static const OperandModifier operand_modifier = modifier;
  FLAT_MSG_DEFINE_OPTIONAL(MirroredObjectOperand, operand);
FLAT_MSG_END(LeveledMutableMirroredObjectOperand);

using ConstMirroredObjectOperand = LeveledMutableMirroredObjectOperand<kConstModifier>;
using MutableMirroredObjectOperand = LeveledMutableMirroredObjectOperand<kDataMutableModifier>;
using Mut2MirroredObjectOperand = LeveledMutableMirroredObjectOperand<kTypeAndDataMutableModifier>;

class InstructionOperandProto;

FLAT_MSG_BEGIN(InstructionOperand);
  // methods
  PUBLIC void __Init__(const InstructionOperandProto& proto);
  // fields
  FLAT_MSG_DEFINE_STRICT_ONEOF(_,
    FLAT_MSG_ONEOF_FIELD(ConstMirroredObjectOperand, const_operand)
    FLAT_MSG_ONEOF_FIELD(MutableMirroredObjectOperand, mutable_operand)
    FLAT_MSG_ONEOF_FIELD(Mut2MirroredObjectOperand, mut2_operand)
    FLAT_MSG_ONEOF_FIELD(double, double_i_operand) // i is short for immediate
    FLAT_MSG_ONEOF_FIELD(int64_t, int64_i_operand)
    FLAT_MSG_ONEOF_FIELD(uint64_t, uint64_i_operand)
    FLAT_MSG_ONEOF_FIELD(bool, bool_i_operand));
FLAT_MSG_END(InstructionOperand);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_INSTRUCTION_OPERAND_H_
