#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/vm.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream_type.h"

namespace oneflow {
namespace vm {

ObjectMsgPtr<InstructionMsg> NewInstruction(const std::string& instr_type_name) {
  return ObjectMsgPtr<InstructionMsg>::New(LookupInstrTypeId(instr_type_name));
}

Maybe<void> Run(const InstructionListProto& instruction_list_proto) {
  TODO();
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
