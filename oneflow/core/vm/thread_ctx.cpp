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
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void ThreadCtx::LoopRun(const std::function<void(ThreadCtx*)>& Initializer) {
  Initializer(this);
  while (ReceiveAndRun() == intrusive::kChannelStatusSuccess) {}
}

intrusive::ChannelStatus ThreadCtx::ReceiveAndRun() {
  const StreamType& stream_type = stream_rt_desc().stream_type();
  intrusive::List<INTRUSIVE_FIELD(Instruction, pending_instruction_hook_)> tmp_list;
  intrusive::ChannelStatus status = mut_pending_instruction_list()->MoveTo(&tmp_list);
  INTRUSIVE_FOR_EACH(instruction, &tmp_list) {
    tmp_list.Erase(instruction.Mutable());
    stream_type.Run(instruction.Mutable());
  }
  return status;
}

intrusive::ChannelStatus ThreadCtx::TryReceiveAndRun() {
  const StreamType& stream_type = stream_rt_desc().stream_type();
  intrusive::List<INTRUSIVE_FIELD(Instruction, pending_instruction_hook_)> tmp_list;
  intrusive::ChannelStatus status = mut_pending_instruction_list()->TryMoveTo(&tmp_list);
  INTRUSIVE_FOR_EACH_PTR(instruction, &tmp_list) {
    CHECK_GT(instruction->ref_cnt(), 1);
    tmp_list.Erase(instruction);
    stream_type.Run(instruction);
  }
  return status;
}

}  // namespace vm
}  // namespace oneflow
