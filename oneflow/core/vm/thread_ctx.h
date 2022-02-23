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
#include "oneflow/core/intrusive/channel.h"
#include "oneflow/core/vm/stream.h"

namespace oneflow {
namespace vm {

using PendingInstructionChannel =
    intrusive::Channel<INTRUSIVE_FIELD(Instruction, pending_instruction_hook_)>;
using PendingInstructionList =
    intrusive::List<INTRUSIVE_FIELD(Instruction, pending_instruction_hook_)>;

class ThreadCtx final : public intrusive::Base {
 public:
  // types
  using StreamList = intrusive::List<INTRUSIVE_FIELD(Stream, thread_ctx_stream_hook_)>;

  // Getters
  const StreamList& stream_list() const { return stream_list_; }

  // Setters
  StreamList* mut_stream_list() { return &stream_list_; }
  PendingInstructionChannel* mut_pending_instruction_list() { return &pending_instruction_list_; }

  // methods
  size_t TryReceiveAndRun();
  intrusive::ChannelStatus ReceiveAndRun();

 private:
  template<intrusive::ChannelStatus (PendingInstructionChannel::*Move)(PendingInstructionList*)>
  intrusive::ChannelStatus MoveAndRun(size_t* cnt);

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  ThreadCtx() : intrusive_ref_(), stream_list_(), pending_instruction_list_(), thread_ctx_hook_() {}
  intrusive::Ref intrusive_ref_;
  // lists
  StreamList stream_list_;
  PendingInstructionChannel pending_instruction_list_;

 public:
  // list hooks
  intrusive::ListHook thread_ctx_hook_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD__H_
