#include "oneflow/core/vm/vm_instruction_id.msg.h"
#include "oneflow/core/vm/vm_stream_type.h"

namespace oneflow {

void VmInstructionId::__Init__(const std::string& vm_instr_type_name) {
  CopyFrom(LookupVmInstructionId(vm_instr_type_name));
}

}  // namespace oneflow
