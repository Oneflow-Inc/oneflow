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

#include "oneflow/core/vm/stream_desc.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {
namespace vm {

class ThreadCtx;

class Stream final : public intrusive::Base {
 public:
  // types
  using DispatchedInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, dispatched_instruction_hook_)>;

  // Getters
  int64_t max_device_num_per_machine() const { return max_device_num_per_machine_; }
  const ThreadCtx& thread_ctx() const { return *thread_ctx_; }
  bool has_thread_ctx() const { return thread_ctx_ != nullptr; }
  const std::unique_ptr<DeviceCtx>& device_ctx() const { return device_ctx_; }
  const intrusive::ListHook& active_stream_hook() const { return active_stream_hook_; }
  const DispatchedInstructionList& free_instruction_list() const { return free_instruction_list_; }
  const DispatchedInstructionList& zombie_instruction_list() const {
    return zombie_instruction_list_;
  }
  const DispatchedInstructionList& running_instruction_list() const {
    return running_instruction_list_;
  }
  const StreamId& stream_id() const { return stream_id_.key(); }

  // Setters
  void set_max_device_num_per_machine(int64_t val) { max_device_num_per_machine_ = val; }
  ThreadCtx* mut_thread_ctx() { return thread_ctx_; }
  void set_thread_ctx(ThreadCtx* val) { thread_ctx_ = val; }
  void clear_thread_ctx() { thread_ctx_ = nullptr; }
  std::unique_ptr<DeviceCtx>* mut_device_ctx() { return &device_ctx_; }
  DispatchedInstructionList* mut_free_instruction_list() { return &free_instruction_list_; }
  DispatchedInstructionList* mut_zombie_instruction_list() { return &zombie_instruction_list_; }
  DispatchedInstructionList* mut_running_instruction_list() { return &running_instruction_list_; }
  StreamId* mut_stream_id() { return stream_id_.mut_key(); }

  // methods
  void __Init__();
  void __Init__(ThreadCtx* thread_ctx, const StreamId& stream_id,
                const int64_t max_device_num_per_machine);
  intrusive::shared_ptr<Instruction> NewInstruction(
      InstructionMsg* instr_msg, const std::shared_ptr<const ParallelDesc>& parallel_desc);
  void DeleteInstruction(intrusive::shared_ptr<Instruction>&&);
  int64_t global_device_id() const { return stream_id().global_device_id(); }
  int64_t machine_id() const;
  int64_t device_id() const;
  const StreamType& stream_type() const;

 private:
  void MoveToFreeList(intrusive::shared_ptr<Instruction>&& instruction);
  void MoveFromZombieListToFreeList();

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  Stream()
      : intrusive_ref_(),
        thread_ctx_(),
        device_ctx_(),
        max_device_num_per_machine_(),
        free_instruction_list_(),
        zombie_instruction_list_(),
        running_instruction_list_(),
        stream_id_(),
        active_stream_hook_(),
        thread_ctx_stream_hook_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  ThreadCtx* thread_ctx_;
  std::unique_ptr<DeviceCtx> device_ctx_;
  int64_t max_device_num_per_machine_;
  // lists
  DispatchedInstructionList free_instruction_list_;
  DispatchedInstructionList zombie_instruction_list_;
  DispatchedInstructionList running_instruction_list_;

 public:
  // skiplist hooks
  intrusive::SkipListHook<StreamId, 10> stream_id_;
  // list hooks
  intrusive::ListHook active_stream_hook_;
  intrusive::ListHook thread_ctx_stream_hook_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_H_
