#ifndef ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_

#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/instruction_operand.msg.h"

namespace oneflow {
namespace eager {

// clang-format off
FLAT_MSG_VIEW_BEGIN(NewOpKernelObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::SymbolOperand, job_desc);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::SymbolOperand, op_conf);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutOperand, op);
FLAT_MSG_VIEW_END(NewOpKernelObjectInstrOperand);

FLAT_MSG_VIEW_BEGIN(DeleteOpKernelObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutOperand, op);
FLAT_MSG_VIEW_END(DeleteOpKernelObjectInstrOperand);

FLAT_MSG_VIEW_BEGIN(CallOpKernelInstrOperand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::MutOperand, opkernel);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::SymbolOperand, op_parallel_attribute);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_ibn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::SymbolOperand, ibn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::ConstOperand, input_blob);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::SymbolOperand, obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutOperand, output_blob);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_mut2_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::SymbolOperand, mut2_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::Mut2Operand, mut2_output_blob);
FLAT_MSG_VIEW_END(CallOpKernelInstrOperand);

FLAT_MSG_VIEW_BEGIN(StatelessCallOpKernelInstrOperand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::SymbolOperand, job_desc);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::SymbolOperand, op_conf);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::SymbolOperand, op_parallel_attribute);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::MutOperand, shared_opkernel);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_ibn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::SymbolOperand, ibn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::ConstOperand, input_blob);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::SymbolOperand, obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::MutOperand, output_blob);

  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::OperandSeparator, begin_mut2_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::SymbolOperand, mut2_obn);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(vm::Mut2Operand, mut2_output_blob);
FLAT_MSG_VIEW_END(StatelessCallOpKernelInstrOperand);

FLAT_MSG_VIEW_BEGIN(FetchBlobInstrOperand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::ConstOperand, blob);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, unique_callback_id);
FLAT_MSG_VIEW_END(FetchBlobInstrOperand);

FLAT_MSG_VIEW_BEGIN(FeedBlobInstrOperand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::Mut2Operand, blob);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, unique_callback_id);
FLAT_MSG_VIEW_END(FeedBlobInstrOperand);

FLAT_MSG_VIEW_BEGIN(RemoveForeignCallbackInstrOperand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(vm::MutOperand, object_id);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, unique_callback_id);
FLAT_MSG_VIEW_END(RemoveForeignCallbackInstrOperand);
// clang-format on

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_OPKERNEL_INSTRUCTION_MSG_H_
