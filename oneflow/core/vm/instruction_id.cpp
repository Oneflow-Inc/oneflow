#include "oneflow/core/vm/instruction_id.msg.h"
#include "oneflow/core/vm/stream_type.h"

namespace oneflow {
namespace vm {

void InstructionId::__Init__(const std::string& instr_type_name) {
  CopyFrom(LookupInstructionId(instr_type_name));
}

}  // namespace vm
}  // namespace oneflow
