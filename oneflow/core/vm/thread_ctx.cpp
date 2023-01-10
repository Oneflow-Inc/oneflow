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

ThreadCtx::ThreadCtx()
    : intrusive_ref_(),
      stream_list_(),
      worker_pending_instruction_mutex_(),
      worker_pending_instruction_list_(&worker_pending_instruction_mutex_),
      notifier_(),
      transport_dependence_(intrusive::make_shared<vm::Dependence>()),
      thread_ctx_hook_() {}

size_t ThreadCtx::TryReceiveAndRun() {
  intrusive::List<INTRUSIVE_FIELD(Instruction, worker_pending_instruction_hook_)> tmp_list;
  mut_worker_pending_instruction_list()->MoveTo(&tmp_list);
  size_t size = tmp_list.size();
  INTRUSIVE_FOR_EACH(instruction, &tmp_list) {
    tmp_list.Erase(instruction.Mutable());
    const StreamPolicy& stream_policy = instruction->stream().stream_policy();
    stream_policy.RunIf(instruction.Mutable());
  }
  return size;
}

}  // namespace vm
}  // namespace oneflow
