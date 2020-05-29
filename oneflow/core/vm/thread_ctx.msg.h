#ifndef ONEFLOW_CORE_VM_THREAD_MSG_H_
#define ONEFLOW_CORE_VM_THREAD_MSG_H_

#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/stream_runtime_desc.msg.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(ThreadCtx);
  // methods
  PUBLIC void __Init__(const StreamRtDesc& stream_rt_desc) {
    set_stream_rt_desc(&stream_rt_desc);
  }
  PUBLIC void LoopRun();
  // fields
  OBJECT_MSG_DEFINE_PTR(const StreamRtDesc, stream_rt_desc); 

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(thread_ctx_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(Stream, thread_ctx_stream_link, stream_list);
  OBJECT_MSG_DEFINE_CONDITION_LIST_HEAD(Instruction, pending_instruction_link, pending_instruction_list);

  PRIVATE ObjectMsgConditionListStatus ReceiveAndRun();
  PRIVATE ObjectMsgConditionListStatus TryReceiveAndRun();
OBJECT_MSG_END(ThreadCtx);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD_MSG_H_
