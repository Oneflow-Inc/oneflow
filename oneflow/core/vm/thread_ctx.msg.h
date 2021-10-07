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
#ifndef ONEFLOW_CORE_VM_THREAD_MSG_H_
#define ONEFLOW_CORE_VM_THREAD_MSG_H_

#include <functional>
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/object_msg/channel.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/stream_runtime_desc.msg.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(ThreadCtx);
 public:
  void __Init__() { clear_stream_rt_desc(); }

  // types
  using StreamList = intrusive::List<OBJECT_MSG_FIELD(Stream, thread_ctx_stream_entry_)>;
  using PendingInstructionChannel =
      intrusive::Channel<OBJECT_MSG_FIELD(Instruction, pending_instruction_entry_)>;

  // Getters
  bool has_stream_rt_desc() const { return stream_rt_desc_ != nullptr; }
  const StreamRtDesc& stream_rt_desc() const { return *stream_rt_desc_; }
  const StreamList& stream_list() const { return stream_list_; }

  // Setters
  void set_stream_rt_desc(const StreamRtDesc* val) { stream_rt_desc_ = val; }
  void clear_stream_rt_desc() { stream_rt_desc_ = nullptr; }
  StreamList* mut_stream_list() { return &stream_list_; }
  StreamList* mutable_stream_list() { return &stream_list_; }
  PendingInstructionChannel* mut_pending_instruction_list() { return &pending_instruction_list_; }
  PendingInstructionChannel* mutable_pending_instruction_list() { return &pending_instruction_list_; }

  // methods
  OF_PUBLIC void __Init__(const StreamRtDesc& stream_rt_desc) {
    __Init__();
    set_stream_rt_desc(&stream_rt_desc);
  }
  OF_PUBLIC void LoopRun(const std::function<void(ThreadCtx*)>& Initializer);
  // fields
  OBJECT_MSG_DEFINE_FIELD(const StreamRtDesc*, stream_rt_desc_); 

  // list entries
  OBJECT_MSG_DEFINE_FIELD(intrusive::ListEntry, thread_ctx_entry_);
  OBJECT_MSG_DEFINE_FIELD(StreamList, stream_list_);
  OBJECT_MSG_DEFINE_FIELD(PendingInstructionChannel, pending_instruction_list_);

  OF_PRIVATE intrusive::ChannelStatus ReceiveAndRun();
  OF_PUBLIC intrusive::ChannelStatus TryReceiveAndRun();
OBJECT_MSG_END(ThreadCtx);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD_MSG_H_
