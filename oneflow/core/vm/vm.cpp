#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/vm/vm.h"
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
  if (Global<ResourceDesc>::Get()->TotalMachineNum() > 1) { TODO(); }
  InstructionListProto instruction_list_proto;
  OF_CHECK(TxtString2PbMessage(instruction_list_str, &instruction_list_proto))
      << "InstructionListProto parse failed";
  InstructionMsgList local_instr_msg_list;
  InstructionMsgList remote_instr_msg_list;
  for (const auto& instr_proto : instruction_list_proto.instruction()) {
    auto instr_msg = ObjectMsgPtr<InstructionMsg>::New(instr_proto);
    if (instr_msg->instr_type_id().type() == kRemote) {
      remote_instr_msg_list.EmplaceBack(std::move(instr_msg));
    } else if (instr_msg->instr_type_id().type() == kLocal) {
      local_instr_msg_list.EmplaceBack(std::move(instr_msg));
    } else {
      UNIMPLEMENTED();
    }
  }
  auto* local_scheduler = Global<OneflowVM<vm::kLocal>>::Get()->mut_scheduler();
  auto* remote_scheduler = Global<OneflowVM<vm::kRemote>>::Get()->mut_scheduler();
  local_scheduler->Receive(&local_instr_msg_list);
  remote_scheduler->Receive(&remote_instr_msg_list);
  while (!(local_scheduler->Empty() && remote_scheduler->Empty())) {
    local_scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(local_scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
    remote_scheduler->Schedule();
    OBJECT_MSG_LIST_FOR_EACH(remote_scheduler->mut_thread_ctx_list(), t) { t->TryReceiveAndRun(); }
  }
  return Maybe<void>::Ok();
}

}  // namespace vm
}  // namespace oneflow
