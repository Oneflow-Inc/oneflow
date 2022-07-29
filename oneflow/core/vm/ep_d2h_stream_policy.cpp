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

#include "oneflow/core/vm/ep_d2h_stream_policy.h"
#include <memory>
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

void EpD2HStreamPolicy::InitInstructionStatus(const Stream& stream,
                                              InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(EpOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  EpStreamPolicyBase* ep_stream_policy_base =
      dynamic_cast<EpStreamPolicyBase*>(const_cast<Stream&>(stream).mut_stream_policy());
  CHECK_NOTNULL(ep_stream_policy_base);
  auto* ep_event_provider = ep_stream_policy_base->ep_event_provider();
  auto* data_ptr = status_buffer->mut_buffer();
  const auto& ep_event = CHECK_NOTNULL(ep_event_provider)->GetReusedEpEvent();
  EpOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, ep_event);
}

void EpD2HStreamPolicy::DeleteInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  auto* ptr = EpOptionalEventRecordStatusQuerier::MutCast(status_buffer->mut_buffer());
  ptr->~EpOptionalEventRecordStatusQuerier();
}

bool EpD2HStreamPolicy::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return EpOptionalEventRecordStatusQuerier::Cast(status_buffer.buffer())->done();
}

void EpD2HStreamPolicy::Run(Instruction* instruction) const {
  OF_PROFILER_RANGE_GUARD("S:" + instruction->DebugName());
  auto* stream = instruction->mut_stream();
  EpStreamPolicyBase* ep_stream_policy_base =
      dynamic_cast<EpStreamPolicyBase*>(stream->mut_stream_policy());
  CHECK_NOTNULL(ep_stream_policy_base);
  auto* ep_device = ep_stream_policy_base->GetOrCreateEpDevice();
  ep_device->SetAsActiveDevice();
  instruction->Compute();
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer();
  EpOptionalEventRecordStatusQuerier::MutCast(data_ptr)->SetLaunched(
      stream->mut_stream_policy()->stream());
}

}  // namespace vm
}  // namespace oneflow
