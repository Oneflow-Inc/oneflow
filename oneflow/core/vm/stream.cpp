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
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void Stream::__Init__(ThreadCtx* thread_ctx, const StreamId& stream_id,
                      const int64_t max_device_num_per_machine) {
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

const StreamTypeId& Stream::stream_type_id() const {
  return thread_ctx().stream_rt_desc().stream_type_id();
}

ObjectMsgPtr<Instruction> Stream::NewInstruction(
    InstructionMsg* instr_msg, const std::shared_ptr<ParallelDesc>& parallel_desc) {
  if (free_instruction_list().empty()) {
    return ObjectMsgPtr<Instruction>::NewFrom(mut_allocator(), instr_msg, this, parallel_desc);
  }
  ObjectMsgPtr<Instruction> instruction = mut_free_instruction_list()->PopFront();
  instruction->__Init__(instr_msg, this, parallel_desc);
  return instruction;
}

void Stream::DeleteInstruction(ObjectMsgPtr<Instruction>&& instruction) {
  CHECK(instruction->is_pending_instruction_link_empty());
  CHECK(instruction->is_instruction_link_empty());
  CHECK_EQ(instruction->ref_cnt(), 1);
  auto* instruction_ptr = instruction.Mutable();
  mut_free_instruction_list()->EmplaceBack(std::move(instruction));
  instruction_ptr->__Delete__();
}

}  // namespace vm
}  // namespace oneflow
