#ifndef ONEFLOW_CORE_VM_INSTRUCTION_OPERAND_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_OPERAND_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/logical_object_id.h"
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
FLAT_MSG_BEGIN(ModifiedMirroredObjectOperand);
  PUBLIC static const OperandModifier operand_modifier = modifier;
  // methods
  PUBLIC int64_t logical_object_id() const { return operand().logical_object_id(); }
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(MirroredObjectOperand, operand);
FLAT_MSG_END(ModifiedMirroredObjectOperand);

using ConstMirroredObjectOperand = ModifiedMirroredObjectOperand<kConstModifier>;
using MutableMirroredObjectOperand = ModifiedMirroredObjectOperand<kDataMutableModifier>;
using Mut2MirroredObjectOperand = ModifiedMirroredObjectOperand<kTypeAndDataMutableModifier>;

FLAT_MSG_BEGIN(OperandSeparator);
FLAT_MSG_END(OperandSeparator);

class InstructionOperandProto;

FLAT_MSG_BEGIN(InstructionOperand);
  // methods
  PUBLIC void __Init__(const InstructionOperandProto& proto);
  // fields
  FLAT_MSG_DEFINE_STRICT_ONEOF(_,
    FLAT_MSG_ONEOF_FIELD(ConstMirroredObjectOperand, const_operand)
    FLAT_MSG_ONEOF_FIELD(MutableMirroredObjectOperand, mutable_operand)
    FLAT_MSG_ONEOF_FIELD(Mut2MirroredObjectOperand, mut2_operand)
    FLAT_MSG_ONEOF_FIELD(OperandSeparator, sep)
    FLAT_MSG_ONEOF_FIELD(double, double_i_operand) // i is short for immediate
    FLAT_MSG_ONEOF_FIELD(int64_t, int64_i_operand)
    FLAT_MSG_ONEOF_FIELD(uint64_t, uint64_i_operand)
    FLAT_MSG_ONEOF_FIELD(bool, bool_i_operand));
FLAT_MSG_END(InstructionOperand);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_OPERAND_H_
