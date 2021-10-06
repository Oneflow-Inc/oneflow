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

struct ThreadCtx;

// clang-format off
OBJECT_MSG_BEGIN(Stream);
 public:
  void __Init__();

  // Getters
  int64_t max_device_num_per_machine() const { return max_device_num_per_machine_; }
  const ThreadCtx& thread_ctx() const { return *thread_ctx_; }
  bool has_thread_ctx() const { return thread_ctx_ != nullptr; }
  const std::unique_ptr<DeviceCtx>& device_ctx() const { return device_ctx_; }
  // Setters
  void set_max_device_num_per_machine(int64_t val) { max_device_num_per_machine_ = val; }
  ThreadCtx* mut_thread_ctx() { return thread_ctx_; }
  ThreadCtx* mutable_thread_ctx() { return thread_ctx_; }
  void set_thread_ctx(ThreadCtx* val) { thread_ctx_ = val; }
  void clear_thread_ctx() { thread_ctx_ = nullptr; }
  std::unique_ptr<DeviceCtx>* mut_device_ctx() { return &device_ctx_; }
  std::unique_ptr<DeviceCtx>* mutable_device_ctx() { return &device_ctx_; }

  // methods
  OF_PUBLIC void __Init__(ThreadCtx* thread_ctx, const StreamId& stream_id, const int64_t max_device_num_per_machine);
  OF_PUBLIC ObjectMsgPtr<Instruction> NewInstruction(InstructionMsg* instr_msg, const std::shared_ptr<const ParallelDesc>& parallel_desc);
  OF_PUBLIC void DeleteInstruction(ObjectMsgPtr<Instruction>&&);
  OF_PUBLIC int64_t global_device_id() const { return stream_id().global_device_id(); }
  OF_PUBLIC int64_t machine_id() const;
  OF_PUBLIC int64_t device_id() const;
  OF_PUBLIC const StreamType& stream_type() const;
  OF_PUBLIC const StreamTypeId& stream_type_id() const;
  OF_PRIVATE void MoveToFreeList(ObjectMsgPtr<Instruction>&& instruction);
  OF_PRIVATE void MoveFromZombieListToFreeList();

  // fields
  OBJECT_MSG_FIELD(ThreadCtx*, thread_ctx_); 
  OBJECT_MSG_FIELD(std::unique_ptr<DeviceCtx>, device_ctx_);
  OBJECT_MSG_FIELD(int64_t, max_device_num_per_machine_);
  
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(active_stream_link);
  OBJECT_MSG_DEFINE_LIST_LINK(thread_ctx_stream_link);
  OBJECT_MSG_DEFINE_MAP_KEY(StreamId, stream_id);

  // heads 
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, free_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, zombie_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, running_instruction_list);
OBJECT_MSG_END(Stream);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_H_
