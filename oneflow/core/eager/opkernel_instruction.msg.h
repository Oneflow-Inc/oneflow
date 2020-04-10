#ifndef ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_

#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/instruction_operand.msg.h"

namespace oneflow {
namespace eager {

// clang-format off
FLAT_MSG_VIEW_BEGIN(NewOpKernelObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::ConstOperand, job);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutOperand, op);
FLAT_MSG_VIEW_END(NewOpKernelObjectInstrOperand);

FLAT_MSG_VIEW_BEGIN(DeleteOpKernelObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutOperand, op);
FLAT_MSG_VIEW_END(DeleteOpKernelObjectInstrOperand);

FLAT_MSG_VIEW_BEGIN(CallOpKernelInstrOperand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::MutOperand, opkernel);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_ibn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::ConstOperand, ibn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, input_index);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::ConstOperand, input_blob);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::ConstOperand, obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, output_index);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutOperand, output_blob);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_mut2_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::ConstOperand, mut2_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int64_t, mut2_output_index);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::Mut2Operand, mut2_output_blob);
FLAT_MSG_VIEW_END(CallOpKernelInstrOperand);
// clang-format on

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_
