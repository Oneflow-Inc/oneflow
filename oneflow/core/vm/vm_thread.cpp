#include "oneflow/core/vm/vm_thread.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmThread::LoopRun() {
  while (WaitAndRun() == kObjectMsgConditionListStatusSuccess)
    ;
}

ObjectMsgConditionListStatus VmThread::WaitAndRun() {
  const VmStreamType& vm_stream_type = vm_stream_rt_desc().vm_stream_type();
  OBJECT_MSG_LIST(VmInstrChainPackage, pending_pkg_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_pkg_list()->MoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, vm_instr_chain_pkg) {
    vm_stream_type.Run(vm_instr_chain_pkg);
    tmp_list.Erase(vm_instr_chain_pkg);
  }
  return status;
}

}  // namespace oneflow
