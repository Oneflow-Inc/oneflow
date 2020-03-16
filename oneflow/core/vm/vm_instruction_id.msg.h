#ifndef ONEFLOW_CORE_VM_VM_INSTRUCTION_ID_MSG_H_
#define ONEFLOW_CORE_VM_VM_INSTRUCTION_ID_MSG_H_

#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

enum VmType { kVmRemote = 0, kVmLocal };

// clang-format off
FLAT_MSG_BEGIN(VmInstructionId);
  FLAT_MSG_DEFINE_OPTIONAL(VmStreamTypeId, vm_stream_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(VmInstructionOpcode, opcode);
  FLAT_MSG_DEFINE_OPTIONAL(VmType, vm_type);
FLAT_MSG_END(VmInstructionId);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_INSTRUCTION_ID_MSG_H_
