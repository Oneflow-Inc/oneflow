#include "oneflow/core/vm/thread.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void Thread::LoopRun() {
  while (ReceiveAndRun() == kObjectMsgConditionListStatusSuccess)
    ;
}

ObjectMsgConditionListStatus Thread::ReceiveAndRun() {
  const StreamType& stream_type = stream_rt_desc().stream_type();
  OBJECT_MSG_LIST(InstrChain, pending_chain_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_chain_list()->MoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, instr_chain) {
    stream_type.Run(instr_chain);
    tmp_list.Erase(instr_chain);
  }
  return status;
}

ObjectMsgConditionListStatus Thread::TryReceiveAndRun() {
  const StreamType& stream_type = stream_rt_desc().stream_type();
  OBJECT_MSG_LIST(InstrChain, pending_chain_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_chain_list()->TryMoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, instr_chain) {
    stream_type.Run(instr_chain);
    tmp_list.Erase(instr_chain);
  }
  return status;
}

}  // namespace vm
}  // namespace oneflow
