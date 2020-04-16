#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void ThreadCtx::LoopRun() {
  while (ReceiveAndRun() == kObjectMsgConditionListStatusSuccess)
    ;
}

ObjectMsgConditionListStatus ThreadCtx::ReceiveAndRun() {
  const StreamType& stream_type = stream_rt_desc().stream_type();
  OBJECT_MSG_LIST(Instruction, pending_instruction_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_instruction_list()->MoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, instruction) {
    stream_type.Run(instruction);
    tmp_list.Erase(instruction);
  }
  return status;
}

ObjectMsgConditionListStatus ThreadCtx::TryReceiveAndRun() {
  const StreamType& stream_type = stream_rt_desc().stream_type();
  OBJECT_MSG_LIST(Instruction, pending_instruction_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_instruction_list()->TryMoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, instruction) {
    stream_type.Run(instruction);
    tmp_list.Erase(instruction);
  }
  return status;
}

}  // namespace vm
}  // namespace oneflow
