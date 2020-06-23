#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {
namespace vm {

using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

ObjectMsgPtr<InstructionMsg> NewInstruction(const std::string& instr_type_name) {
  return ObjectMsgPtr<InstructionMsg>::New(instr_type_name);
}

Maybe<void> Run(const std::string& instruction_list_str) {
  InstructionListProto instruction_list_proto;
  CHECK_OR_RETURN(TxtString2PbMessage(instruction_list_str, &instruction_list_proto))
      << "InstructionListProto parse failed";
  return Run(instruction_list_proto);
}

Maybe<void> Run(const InstructionListProto& instruction_list_proto) {
  if (JUST(GlobalMaybe<ResourceDesc, ForSession>())->TotalMachineNum() > 1) { TODO(); }
  InstructionMsgList instr_msg_list;
  for (const auto& instr_proto : instruction_list_proto.instruction()) {
    auto instr_msg = ObjectMsgPtr<InstructionMsg>::New(instr_proto);
    instr_msg_list.EmplaceBack(std::move(instr_msg));
  }
  auto* vm = JUST(GlobalMaybe<OneflowVM>())->mut_vm();
  vm->Receive(&instr_msg_list);
  while (!vm->Empty()) {
    vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
