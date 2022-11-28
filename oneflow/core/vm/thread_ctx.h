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
#ifndef ONEFLOW_CORE_VM_THREAD__H_
#define ONEFLOW_CORE_VM_THREAD__H_

#include <functional>
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/intrusive/mutexed_list.h"
#include "oneflow/core/common/notifier.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/vm_object.h"

namespace oneflow {
namespace vm {

using WorkerPendingInstructionMutexedList =
    intrusive::MutexedList<INTRUSIVE_FIELD(Instruction, worker_pending_instruction_hook_)>;

class ThreadCtx final : public intrusive::Base {
 public:
  // types
  using StreamList = intrusive::List<INTRUSIVE_FIELD(Stream, thread_ctx_stream_hook_)>;

  // Getters
  const StreamList& stream_list() const { return stream_list_; }

  // Setters
  StreamList* mut_stream_list() { return &stream_list_; }
  WorkerPendingInstructionMutexedList* mut_worker_pending_instruction_list() {
    return &worker_pending_instruction_list_;
  }

  // methods
  size_t TryReceiveAndRun();

  Notifier* mut_notifier() { return &notifier_; }

  const intrusive::shared_ptr<vm::Dependence>& transport_dependence() const {
    return transport_dependence_;
  };

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  ThreadCtx();

  intrusive::Ref intrusive_ref_;
  // lists
  StreamList stream_list_;
  std::mutex worker_pending_instruction_mutex_;
  WorkerPendingInstructionMutexedList worker_pending_instruction_list_;
  Notifier notifier_;
  intrusive::shared_ptr<vm::Dependence> transport_dependence_;

 public:
  // list hooks
  intrusive::ListHook thread_ctx_hook_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD__H_
