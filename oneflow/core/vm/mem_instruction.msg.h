#ifndef ONEFLOW_CORE_VM_MEM_INSTRUCTION_H_
#define ONEFLOW_CORE_VM_MEM_INSTRUCTION_H_

#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/instruction_operand.msg.h"

namespace oneflow {
namespace vm {

// clang-format off
FLAT_MSG_VIEW_BEGIN(MallocInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutOperand, mem_buffer);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, size);
FLAT_MSG_VIEW_END(MallocInstruction);
// clang-format on

// clang-format off
FLAT_MSG_VIEW_BEGIN(FreeInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutOperand, mem_buffer);
FLAT_MSG_VIEW_END(FreeInstruction);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_INSTRUCTION_H_
