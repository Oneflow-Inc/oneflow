#ifndef ONEFLOW_CORE_VM_VM_INSTRUCTION_ID_MSG_H_
#define ONEFLOW_CORE_VM_VM_INSTRUCTION_ID_MSG_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/vm_stream_desc.msg.h"

namespace oneflow {
namespace vm {

enum VmType { kRemote = 0, kLocal };

// clang-format off
FLAT_MSG_BEGIN(InstructionId);
  PUBLIC void __Init__(const std::string& instr_type_name);

  FLAT_MSG_DEFINE_OPTIONAL(StreamTypeId, vm_stream_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(InstructionOpcode, opcode);
  FLAT_MSG_DEFINE_OPTIONAL(VmType, vm_type);
FLAT_MSG_END(InstructionId);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_INSTRUCTION_ID_MSG_H_
