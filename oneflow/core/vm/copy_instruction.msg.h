#ifndef ONEFLOW_CORE_VM_COPY_INSTRUCTION_MSG_H_
#define ONEFLOW_CORE_VM_COPY_INSTRUCTION_MSG_H_

#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/instruction_operand.msg.h"

namespace oneflow {
namespace vm {

// clang-format off
FLAT_MSG_VIEW_BEGIN(CopyInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutOperand, dst);
  FLAT_MSG_VIEW_DEFINE_PATTERN(ConstOperand, src);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, size);
FLAT_MSG_VIEW_END(CopyInstruction);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_COPY_INSTRUCTION_MSG_H_
