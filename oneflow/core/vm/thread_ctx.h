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
#include "oneflow/core/vm/stream_runtime_desc.h"

namespace oneflow {
namespace vm {

// clang-format off
INTRUSIVE_BEGIN(ThreadCtx);
 public:
  void __Init__() { clear_stream_rt_desc(); }

  // types
  using StreamList = intrusive::List<INTRUSIVE_FIELD(Stream, thread_ctx_stream_hook_)>;
  using PendingInstructionChannel =
      intrusive::Channel<INTRUSIVE_FIELD(Instruction, pending_instruction_hook_)>;

  // Getters
  bool has_stream_rt_desc() const { return stream_rt_desc_ != nullptr; }
  const StreamRtDesc& stream_rt_desc() const { return *stream_rt_desc_; }
  const StreamList& stream_list() const { return stream_list_; }

  // Setters
  void set_stream_rt_desc(const StreamRtDesc* val) { stream_rt_desc_ = val; }
  void clear_stream_rt_desc() { stream_rt_desc_ = nullptr; }
  StreamList* mut_stream_list() { return &stream_list_; }
  PendingInstructionChannel* mut_pending_instruction_list() { return &pending_instruction_list_; }

  // methods
  void __Init__(const StreamRtDesc& stream_rt_desc) {
    __Init__();
    set_stream_rt_desc(&stream_rt_desc);
  }
  void LoopRun(const std::function<void(ThreadCtx*)>& Initializer);
  intrusive::ChannelStatus TryReceiveAndRun();
  
 private:
  intrusive::ChannelStatus ReceiveAndRun();

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  ThreadCtx() : intrusive_ref_(), stream_rt_desc_(), thread_ctx_hook_(), stream_list_(), pending_instruction_list_() {}
  INTRUSIVE_DEFINE_FIELD(intrusive::Ref, intrusive_ref_);
  // fields
  INTRUSIVE_DEFINE_FIELD(const StreamRtDesc*, stream_rt_desc_); 
  // list hooks
  INTRUSIVE_DEFINE_FIELD(intrusive::ListHook, thread_ctx_hook_);
  // lists
  INTRUSIVE_DEFINE_FIELD(StreamList, stream_list_);
  INTRUSIVE_DEFINE_FIELD(PendingInstructionChannel, pending_instruction_list_);
INTRUSIVE_END(ThreadCtx);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_THREAD__H_
