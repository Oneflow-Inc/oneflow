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
#include "oneflow/core/framework/device.h"
#include "oneflow/core/vm/stream_get_stream_type.h"

namespace oneflow {
namespace vm {

void Stream::__Init__(
    ThreadCtx* thread_ctx, Symbol<Device> device, StreamRole stream_role,
    const intrusive::shared_ptr<MirroredObject>& schedule_local_dep_object,
    const Optional<intrusive::shared_ptr<MirroredObject>>& transport_local_dep_object) {
  set_thread_ctx(thread_ctx);
  device_ = device;
  stream_role_ = stream_role;
  stream_type_ = CHECK_JUST(GetStreamType::Visit(stream_role, device->enum_type()));
  stream_type_->InitDeviceCtx(mut_device_ctx(), this);
  schedule_local_dep_object_ = schedule_local_dep_object;
  transport_local_dep_object_ = transport_local_dep_object;
}

int64_t Stream::device_id() const { return device_->device_id(); }

const StreamType& Stream::stream_type() const { return *stream_type_; }

intrusive::shared_ptr<Instruction> Stream::NewInstruction(InstructionMsg* instr_msg) {
  intrusive::shared_ptr<Instruction> instruction;
  if (unlikely(free_instruction_list().empty())) {
    instruction = intrusive::make_shared<Instruction>();
  } else {
    instruction = mut_free_instruction_list()->PopFront();
  }
  instruction->Init(instr_msg);
  return instruction;
}

void Stream::MoveToFreeList(intrusive::shared_ptr<Instruction>&& instruction) {
  CHECK_EQ(instruction->ref_cnt(), 1);
  auto* instruction_ptr = instruction.Mutable();
  mut_free_instruction_list()->EmplaceBack(std::move(instruction));
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
  instruction->Delete();
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
