#ifndef ONEFLOW_CORE_VM_INSTRUCTION_ID_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_ID_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/vm_type.h"

namespace oneflow {
namespace vm {

// clang-format off
FLAT_MSG_BEGIN(InstrTypeId);
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(StreamTypeId, stream_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(InstructionOpcode, opcode);
  FLAT_MSG_DEFINE_OPTIONAL(VmType, type);
FLAT_MSG_END(InstrTypeId);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_ID_H_
