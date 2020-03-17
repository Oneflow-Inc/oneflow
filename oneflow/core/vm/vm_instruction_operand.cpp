#include "oneflow/core/vm/vm_instruction_operand.msg.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void VmInstructionOperand::__Init__(const InstructionOperandProto& proto) {
  if (proto.has_const_operand()) {
    mutable_const_operand()->mutable_operand()->__Init__(proto.const_operand());
  } else if (proto.has_mutable_operand()) {
    mutable_mutable_operand()->mutable_operand()->__Init__(proto.const_operand());
  } else if (proto.has_double_i_operand()) {
    set_double_i_operand(proto.double_i_operand());
  } else if (proto.has_int64_i_operand()) {
    set_int64_i_operand(proto.int64_i_operand());
  } else if (proto.has_uint64_i_operand()) {
    set_uint64_i_operand(proto.uint64_i_operand());
  } else if (proto.has_bool_i_operand()) {
    set_bool_i_operand(proto.bool_i_operand());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace vm
}  // namespace oneflow
