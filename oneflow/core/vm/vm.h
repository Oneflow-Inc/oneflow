#ifndef ONEFLOW_CORE_VM_VM_H_
#define ONEFLOW_CORE_VM_VM_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/object_msg.h"

namespace oneflow {
namespace vm {

class InstructionListProto;
class InstructionMsg;

ObjectMsgPtr<InstructionMsg> NewInstruction(const std::string& instr_type_name);

Maybe<void> Run(const InstructionListProto& instruction_list_proto);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_H_
