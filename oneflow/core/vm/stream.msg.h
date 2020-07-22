#ifndef ONEFLOW_CORE_VM_STREAM_H_
#define ONEFLOW_CORE_VM_STREAM_H_

#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {
namespace vm {

class ThreadCtx;

// clang-format off
OBJECT_MSG_BEGIN(Stream);
  // methods
  OF_PUBLIC void __Init__(ThreadCtx* thread_ctx, const StreamId& stream_id);
  OF_PUBLIC ObjectMsgPtr<Instruction> NewInstruction(InstructionMsg* instr_msg, const std::shared_ptr<ParallelDesc>& parallel_desc);
  OF_PUBLIC void DeleteInstruction(ObjectMsgPtr<Instruction>&&);
  OF_PUBLIC int64_t global_device_id() const { return stream_id().global_device_id(); }
  OF_PUBLIC int64_t machine_id() const;
  OF_PUBLIC int64_t device_id() const;
  OF_PUBLIC const StreamType& stream_type() const;
  OF_PUBLIC const StreamTypeId& stream_type_id() const;

  // fields
  OBJECT_MSG_DEFINE_PTR(ThreadCtx, thread_ctx); 
  OBJECT_MSG_DEFINE_STRUCT(std::unique_ptr<DeviceCtx>, device_ctx);
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(active_stream_link);
  OBJECT_MSG_DEFINE_LIST_LINK(thread_ctx_stream_link);
  OBJECT_MSG_DEFINE_MAP_KEY(StreamId, stream_id);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, running_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, free_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(CallbackMsg, callback_link, callback_list);
OBJECT_MSG_END(Stream);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_H_
