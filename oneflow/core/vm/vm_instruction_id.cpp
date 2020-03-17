#include "oneflow/core/vm/vm_instruction_id.msg.h"
#include "oneflow/core/vm/vm_stream_type.h"

namespace oneflow {
namespace vm {

void VmInstructionId::__Init__(const std::string& instr_type_name) {
  CopyFrom(LookupVmInstructionId(instr_type_name));
}

}  // namespace vm
}  // namespace oneflow
