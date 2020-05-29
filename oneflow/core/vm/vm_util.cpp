#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction.pb.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/job/resource_desc.h"

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
  if (JUST(GlobalMaybe<ResourceDesc>())->TotalMachineNum() > 1) { TODO(); }
  InstructionMsgList local_instr_msg_list;
  InstructionMsgList remote_instr_msg_list;
  for (const auto& instr_proto : instruction_list_proto.instruction()) {
    auto instr_msg = ObjectMsgPtr<InstructionMsg>::New(instr_proto);
    if (instr_msg->instr_type_id().type() == kWorker) {
      remote_instr_msg_list.EmplaceBack(std::move(instr_msg));
    } else if (instr_msg->instr_type_id().type() == kMaster) {
      local_instr_msg_list.EmplaceBack(std::move(instr_msg));
    } else {
      UNIMPLEMENTED();
    }
  }
  auto* local_vm = JUST(GlobalMaybe<OneflowVM<vm::kMaster>>())->mut_vm();
  auto* remote_vm = JUST(GlobalMaybe<OneflowVM<vm::kWorker>>())->mut_vm();
  remote_vm->Receive(&remote_instr_msg_list);
  while (!(local_vm->Empty() && remote_vm->Empty())) {
    local_vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(local_vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    remote_vm->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(remote_vm->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
