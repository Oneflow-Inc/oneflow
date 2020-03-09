#ifndef ONEFLOW_CORE_VM_VM_INSTRUCTION_OPERAND_H_
#define ONEFLOW_CORE_VM_VM_INSTRUCTION_OPERAND_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"
#include "oneflow/core/vm/mirrored_object_id.msg.h"

namespace oneflow {

// clang-format off
FLAT_MSG_BEGIN(ConstMirroredObjectOperand);
  FLAT_MSG_DEFINE_OPTIONAL(MirroredObjectOperand, operand);
FLAT_MSG_END(ConstMirroredObjectOperand);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(MutableMirroredObjectOperand);
  FLAT_MSG_DEFINE_OPTIONAL(MirroredObjectOperand, operand);
FLAT_MSG_END(MutableMirroredObjectOperand);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(ConstLocalMirroredObjectOperand);
  FLAT_MSG_DEFINE_OPTIONAL(MirroredObjectOperand, operand);
FLAT_MSG_END(ConstLocalMirroredObjectOperand);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(MutableLocalMirroredObjectOperand);
  FLAT_MSG_DEFINE_OPTIONAL(MirroredObjectOperand, operand);
FLAT_MSG_END(MutableLocalMirroredObjectOperand);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(VmInstructionOperand);
  FLAT_MSG_DEFINE_STRICT_ONEOF(_,
    FLAT_MSG_ONEOF_FIELD(ConstMirroredObjectOperand, const_operand)
    FLAT_MSG_ONEOF_FIELD(MutableMirroredObjectOperand, mutable_operand)
    FLAT_MSG_ONEOF_FIELD(ConstLocalMirroredObjectOperand, const_local_operand)
    FLAT_MSG_ONEOF_FIELD(MutableLocalMirroredObjectOperand, mutable_local_operand)
    FLAT_MSG_ONEOF_FIELD(double, double_i_operand) // i is short for immediate
    FLAT_MSG_ONEOF_FIELD(int64_t, int64_i_operand)
    FLAT_MSG_ONEOF_FIELD(uint64_t, uint64_i_operand)
    FLAT_MSG_ONEOF_FIELD(bool, bool_i_operand));
FLAT_MSG_END(VmInstructionOperand);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_INSTRUCTION_OPERAND_H_
