#ifndef ONEFLOW_CORE_VM_VM_STREAM_H_
#define ONEFLOW_CORE_VM_VM_STREAM_H_

#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {
namespace vm {

class ThreadCtx;

// clang-format off
OBJECT_MSG_BEGIN(Stream);
  // methods
  PUBLIC void __Init__(ThreadCtx* thread_ctx, const StreamId& stream_id);
  PUBLIC ObjectMsgPtr<InstrChain> NewInstrChain(InstructionMsg* instr_msg);
  PUBLIC void DeleteInstrChain(ObjectMsgPtr<InstrChain>&&);
  PUBLIC int64_t parallel_id() const { return stream_id().parallel_id(); }
  PUBLIC int64_t machine_id() const;
  PUBLIC const StreamType& stream_type() const;

  // fields
  OBJECT_MSG_DEFINE_PTR(ThreadCtx, thread_ctx); 
  OBJECT_MSG_DEFINE_STRUCT(std::unique_ptr<DeviceCtx>, device_ctx);
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(active_stream_link);
  OBJECT_MSG_DEFINE_LIST_LINK(thread_ctx_stream_link);
  OBJECT_MSG_DEFINE_MAP_FLAT_MSG_KEY(StreamId, stream_id);
  OBJECT_MSG_DEFINE_LIST_HEAD(InstrChain, instr_chain_link, running_chain_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(InstrChain, instr_chain_link, free_chain_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(CallbackMsg, callback_link, callback_list);
OBJECT_MSG_END(Stream);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_H_
