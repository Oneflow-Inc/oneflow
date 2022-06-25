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

#include "oneflow/core/vm/lazy_job_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/lazy_job_device_context.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void LazyJobStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                      Stream* stream) const {
  device_ctx->reset(new LazyJobDeviceCtx());
}

void LazyJobStreamType::InitInstructionStatus(const Stream& stream,
                                              InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer());
}

void LazyJobStreamType::DeleteInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  auto* ptr = NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer());
  ptr->~NaiveInstrStatusQuerier();
}

bool LazyJobStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer())->done();
}

void LazyJobStreamType::Compute(Instruction* instruction) const {
  instruction->instruction_type().Compute(instruction);
}

}  // namespace vm
}  // namespace oneflow
