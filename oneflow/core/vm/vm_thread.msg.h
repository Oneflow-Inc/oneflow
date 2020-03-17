#ifndef ONEFLOW_CORE_VM_VM_THREAD_MSG_H_
#define ONEFLOW_CORE_VM_VM_THREAD_MSG_H_

#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/vm_stream_runtime_desc.msg.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(Thread);
  // methods
  PUBLIC void __Init__(const StreamRtDesc& vm_stream_rt_desc, int64_t device_id) {
    set_vm_stream_rt_desc(&vm_stream_rt_desc);
    set_device_id(device_id);
  }
  PUBLIC void LoopRun();
  // fields
  OBJECT_MSG_DEFINE_PTR(const StreamRtDesc, vm_stream_rt_desc); 
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, device_id);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(vm_thread_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(Stream, vm_thread_vm_stream_link, vm_stream_list);
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(InstrChain, pending_chain_link, pending_chain_list);

  PRIVATE ObjectMsgConditionListStatus ReceiveAndRun();
  PRIVATE ObjectMsgConditionListStatus TryReceiveAndRun();
OBJECT_MSG_END(Thread);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_THREAD_MSG_H_
