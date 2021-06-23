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
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void ThreadCtx::LoopRun() {
  while (ReceiveAndRun() == kObjectMsgConditionListStatusSuccess) {}
}

ObjectMsgConditionListStatus ThreadCtx::ReceiveAndRun() {
  const StreamType& stream_type = stream_rt_desc().stream_type();
  OBJECT_MSG_LIST(Instruction, pending_instruction_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_instruction_list()->MoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH(&tmp_list, instruction) {
    tmp_list.Erase(instruction.Mutable());
    stream_type.Run(instruction.Mutable());
  }
  return status;
}

ObjectMsgConditionListStatus ThreadCtx::TryReceiveAndRun() {
  const StreamType& stream_type = stream_rt_desc().stream_type();
  OBJECT_MSG_LIST(Instruction, pending_instruction_link) tmp_list;
  ObjectMsgConditionListStatus status = mut_pending_instruction_list()->TryMoveTo(&tmp_list);
  OBJECT_MSG_LIST_FOR_EACH_PTR(&tmp_list, instruction) {
    CHECK_GT(instruction->ref_cnt(), 1);
    tmp_list.Erase(instruction);
    stream_type.Run(instruction);
  }
  return status;
}

}  // namespace vm
}  // namespace oneflow
