#ifndef ONEFLOW_CORE_VM_VM_STREAM_H_
#define ONEFLOW_CORE_VM_VM_STREAM_H_

#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/vm_instruction.msg.h"

namespace oneflow {

class VmThread;

// clang-format off
BEGIN_OBJECT_MSG(VmStream);
  // methods
  PUBLIC void __Init__(VmThread* vm_thread, const VmStreamId& vm_stream_id) {
    set_vm_thread(vm_thread);
    mut_vm_stream_id()->CopyFrom(vm_stream_id);
  }
  PUBLIC ObjectMsgPtr<VmInstrChainPackage> NewVmInstrChainPackage();
  PUBLIC void DeleteVmInstrChainPackage(VmInstrChainPackage*);

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(VmThread, vm_thread); 
  OBJECT_MSG_DEFINE_FLAT_MSG(VmStreamId, vm_stream_id);
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(active_vm_stream_link);
  OBJECT_MSG_DEFINE_LIST_LINK(tmp_active_vm_stream_link);
  OBJECT_MSG_DEFINE_LIST_LINK(vm_thread_vm_stream_link);
  OBJECT_MSG_DEFINE_MAP_KEY(int64_t, parallel_id);
  // collect_vm_instr_chain_list used by VmScheduler
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstrChain, vm_instr_chain_link,
                              collect_vm_instr_chain_list);
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(VmInstrChainPackage, waiting_pkg_link, waiting_pkg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstrChainPackage, vm_instr_chain_pkg_link, running_pkg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstrChainPackage, vm_instr_chain_pkg_link, free_pkg_list);
END_OBJECT_MSG(VmStream);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_H_
