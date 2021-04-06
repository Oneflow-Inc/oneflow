/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
  OF_PUBLIC void __Init__(ThreadCtx* thread_ctx, const StreamId& stream_id, const int64_t max_device_num_per_machine);
  OF_PUBLIC ObjectMsgPtr<Instruction> NewInstruction(InstructionMsg* instr_msg, const std::shared_ptr<ParallelDesc>& parallel_desc);
  OF_PUBLIC void DeleteInstruction(ObjectMsgPtr<Instruction>&&);
  OF_PUBLIC int64_t global_device_id() const { return stream_id().global_device_id(); }
  OF_PUBLIC int64_t machine_id() const;
  OF_PUBLIC int64_t device_id() const;
  OF_PUBLIC const StreamType& stream_type() const;
  OF_PUBLIC const StreamTypeId& stream_type_id() const;
  OF_PRIVATE void MoveToFreeList(ObjectMsgPtr<Instruction>&& instruction);
  OF_PRIVATE void MoveFromZombieListToFreeList();

  // fields
  OBJECT_MSG_DEFINE_PTR(ThreadCtx, thread_ctx); 
  OBJECT_MSG_DEFINE_STRUCT(std::unique_ptr<DeviceCtx>, device_ctx);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, max_device_num_per_machine);
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(active_stream_link);
  OBJECT_MSG_DEFINE_LIST_LINK(thread_ctx_stream_link);
  OBJECT_MSG_DEFINE_MAP_KEY(StreamId, stream_id);

  // heads 
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, running_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, free_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, zombie_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(CallbackMsg, callback_link, callback_list);
OBJECT_MSG_END(Stream);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_H_
