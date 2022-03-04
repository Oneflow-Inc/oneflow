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
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cpp_attribute.h"

namespace oneflow {
namespace vm {

void Stream::__Init__() { clear_thread_ctx(); }

void Stream::__Init__(ThreadCtx* thread_ctx, const StreamId& stream_id,
                      const int64_t max_device_num_per_machine) {
  __Init__();
  set_thread_ctx(thread_ctx);
  mut_stream_id()->CopyFrom(stream_id);
  // InitDeviceCtx may use max_device_num_per_machine,
  // so max_device_num_per_machine must be set before InitDeviceCtx
  set_max_device_num_per_machine(max_device_num_per_machine);
  stream_type().InitDeviceCtx(mut_device_ctx(), this);
}

int64_t Stream::machine_id() const { return global_device_id() / max_device_num_per_machine(); }

int64_t Stream::device_id() const { return global_device_id() % max_device_num_per_machine(); }

const StreamType& Stream::stream_type() const {
  return thread_ctx().stream_rt_desc().stream_type();
}

intrusive::shared_ptr<Instruction> Stream::NewInstruction(
    InstructionMsg* instr_msg, const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  intrusive::shared_ptr<Instruction> instruction;
  if (unlikely(free_instruction_list().empty())) {
    instruction = intrusive::make_shared<Instruction>();
  } else {
    instruction = mut_free_instruction_list()->PopFront();
  }
  instruction->Init(instr_msg, this, parallel_desc);
  return instruction;
}

void Stream::MoveToFreeList(intrusive::shared_ptr<Instruction>&& instruction) {
  CHECK_EQ(instruction->ref_cnt(), 1);
  auto* instruction_ptr = instruction.Mutable();
  mut_free_instruction_list()->EmplaceBack(std::move(instruction));
  instruction_ptr->Delete();
}

void Stream::MoveFromZombieListToFreeList() {
  auto* zombie_list = mut_zombie_instruction_list();
  static const size_t kTryCount = 2;
  for (int i = 0; i < kTryCount; ++i) {
    intrusive::shared_ptr<Instruction> first = zombie_list->Begin();
    if (!first) { break; }
    zombie_list->Erase(first.Mutable());
    size_t ref_cnt = first->ref_cnt();
    if (ref_cnt == 1) {
      MoveToFreeList(std::move(first));
    } else if (ref_cnt == 2) {
      // put `first` back to zombie_list because a worker is holding a reference to `first`
      zombie_list->EmplaceBack(std::move(first));
    } else {
      UNIMPLEMENTED() << "ref_cnt: " << ref_cnt << " first->ref_cnt():" << first->ref_cnt() << "\n"
                      << first->instr_msg().DebugName();
    }
  }
}

void Stream::DeleteInstruction(intrusive::shared_ptr<Instruction>&& instruction) {
  CHECK(instruction->instruction_hook().empty());
  CHECK(instruction->pending_instruction_hook().empty());
  CHECK(instruction->dispatched_instruction_hook().empty());
  // the value of instruction->ref_cnt() may be updated by a worker thread
  size_t ref_cnt = instruction->ref_cnt();
  if (ref_cnt == 1) {
    MoveToFreeList(std::move(instruction));
  } else if (ref_cnt == 2) {
    // a worker is holding a reference to `instruction`
    mut_zombie_instruction_list()->EmplaceBack(std::move(instruction));
  } else {
    UNIMPLEMENTED() << "ref_cnt: " << ref_cnt
                    << " instruction->ref_cnt():" << instruction->ref_cnt() << "\n"
                    << instruction->instr_msg().DebugName();
  }
  MoveFromZombieListToFreeList();
}

}  // namespace vm
}  // namespace oneflow
