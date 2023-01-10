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

#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/vm/stream_policy.h"

namespace oneflow {

class Device;

namespace vm {

class ThreadCtx;
class MirroredObject;
class Dependence;

class Stream final : public intrusive::Base {
 public:
  // types
  using DispatchedInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, dispatched_instruction_hook_)>;

  // Getters
  const StreamPolicy& stream_policy() const { return *stream_policy_; }
  const ThreadCtx& thread_ctx() const { return *thread_ctx_; }
  bool has_thread_ctx() const { return thread_ctx_ != nullptr; }
  const intrusive::ListHook& active_stream_hook() const { return active_stream_hook_; }
  const DispatchedInstructionList& running_instruction_list() const {
    return running_instruction_list_;
  }

  // Setters
  StreamPolicy* mut_stream_policy() { return stream_policy_.get(); }
  ThreadCtx* mut_thread_ctx() { return thread_ctx_; }
  void set_thread_ctx(ThreadCtx* val) { thread_ctx_ = val; }
  void clear_thread_ctx() { thread_ctx_ = nullptr; }
  DispatchedInstructionList* mut_running_instruction_list() { return &running_instruction_list_; }

  // methods
  void __Init__(ThreadCtx* thread_ctx, Symbol<Device> device, StreamType stream_type,
                const intrusive::shared_ptr<Dependence>& schedule_local_dep_object,
                const std::vector<intrusive::shared_ptr<Dependence>>& transport_dependences);
  int64_t device_id() const;
  Symbol<Device> device() const { return device_; }
  StreamType stream_type() const { return stream_type_; }
  bool on_scheduler_thread() const { return on_scheduler_thread_; }

  const intrusive::shared_ptr<Dependence>& schedule_local_dep_object() const {
    return schedule_local_dep_object_;
  }

  const std::vector<intrusive::shared_ptr<Dependence>>& transport_dependences() const {
    return transport_dependences_;
  }

  char* CheckSizeAndGetTmpSmallPinnedMemPtr(size_t size);

 private:
  void MoveToFreeList(intrusive::shared_ptr<Instruction>&& instruction);
  void MoveFromZombieListToFreeList();

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  Stream()
      : intrusive_ref_(),
        thread_ctx_(),
        device_(),
        stream_type_(StreamType::kInvalid),
        stream_policy_(),
        on_scheduler_thread_(false),
        small_pinned_mem_ptr_(),
        running_instruction_list_(),
        active_stream_hook_(),
        thread_ctx_stream_hook_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  ThreadCtx* thread_ctx_;
  Symbol<Device> device_;
  StreamType stream_type_;
  std::shared_ptr<StreamPolicy> stream_policy_;
  bool on_scheduler_thread_;
  std::unique_ptr<char, std::function<void(char*)>> small_pinned_mem_ptr_;
  // lists
  DispatchedInstructionList running_instruction_list_;

  intrusive::shared_ptr<Dependence> schedule_local_dep_object_;
  std::vector<intrusive::shared_ptr<Dependence>> transport_dependences_;

 public:
  // list hooks
  intrusive::ListHook active_stream_hook_;
  intrusive::ListHook thread_ctx_stream_hook_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_H_
