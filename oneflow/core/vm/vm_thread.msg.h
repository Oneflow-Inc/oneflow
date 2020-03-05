#ifndef ONEFLOW_CORE_VM_VM_THREAD_MSG_H_
#define ONEFLOW_CORE_VM_VM_THREAD_MSG_H_

#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/vm_stream_runtime_desc.msg.h"

namespace oneflow {

// clang-format off
OBJECT_MSG_BEGIN(VmThread);
  // methods
  PUBLIC void __Init__(const VmStreamRtDesc& vm_stream_rt_desc) {
    set_vm_stream_rt_desc(&vm_stream_rt_desc);
  }
  PUBLIC void LoopRun();
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const VmStreamRtDesc, vm_stream_rt_desc); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_thread_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmStream, vm_thread_vm_stream_link, vm_stream_list);
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(VmInstrChainPackage, waiting_pkg_link, waiting_pkg_list);

  PRIVATE ObjectMsgConditionListStatus WaitAndRun();
OBJECT_MSG_END(VmThread);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_THREAD_MSG_H_
