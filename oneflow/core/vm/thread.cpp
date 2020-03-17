#include "oneflow/core/vm/thread.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void Thread::LoopRun() {
  while (ReceiveAndRun() == kObjectMsgConditionListStatusSuccess)
    ;
}

ObjectMsgConditionListStatus Thread::ReceiveAndRun() {
  const StreamType& vm_stream_type = vm_stream_rt_desc().vm_stream_type();
  OBJECT_MSG_LIST(InstrChain, pending_chain_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_chain_list()->MoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, vm_instr_chain) {
    vm_stream_type.Run(vm_instr_chain);
    tmp_list.Erase(vm_instr_chain);
  }
  return status;
}

ObjectMsgConditionListStatus Thread::TryReceiveAndRun() {
  const StreamType& vm_stream_type = vm_stream_rt_desc().vm_stream_type();
  OBJECT_MSG_LIST(InstrChain, pending_chain_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_chain_list()->TryMoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, vm_instr_chain) {
    vm_stream_type.Run(vm_instr_chain);
    tmp_list.Erase(vm_instr_chain);
  }
  return status;
}

}  // namespace vm
}  // namespace oneflow
