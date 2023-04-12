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
#include "oneflow/core/vm/stream_record_event_instruction_policy.h"
#include "oneflow/core/vm/ep_event.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/ep/cuda/cuda_event.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/cuda/cuda_device.h"
#include "oneflow/core/vm/ep_stream_policy_base.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"

namespace oneflow {
namespace vm {

StreamRecordEventInstructionPolicy::StreamRecordEventInstructionPolicy(
    const small_vector<intrusive::shared_ptr<LocalDepObject>>& dependences)
    : dependences_(dependences), input_dependences_(), output_dependences_() {
  for (const auto& dep : dependences_) { output_dependences_.push_back(dep.get()); }
}

void StreamRecordEventInstructionPolicy::InitInstructionStatus(Instruction* instruction) {
  auto* stream = instruction->mut_stream();
  {
    auto* ep_stream_policy_base =
        CHECK_NOTNULL(dynamic_cast<EpStreamPolicyBase*>(instruction->mut_stream_policy()));
    ep_stream_policy_base->InitInstructionStatus(*stream, instruction->mut_status_buffer());
    auto* ep_event_provider = ep_stream_policy_base->ep_event_provider();
    const auto& ep_event = CHECK_NOTNULL(ep_event_provider)->GetReusedEpEvent();
    mut_ep_event() = ep_event;
  }
  {
    auto* status_buffer = instruction->mut_status_buffer();
    instruction->stream_policy().InitInstructionStatus(*stream, status_buffer);
    auto* data_ptr = status_buffer->mut_buffer();
    EpOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_ep_event(nullptr);
  }
}

void StreamRecordEventInstructionPolicy::Compute(vm::Instruction* instruction) {
  const auto& ep_event = mut_ep_event();
  // Record event.
  auto* stream_policy =
      dynamic_cast<EpStreamPolicyBase*>(instruction->mut_stream()->mut_stream_policy());
  CHECK_NOTNULL(stream_policy)->stream()->RecordEvent(ep_event->mut_event());
}

}  // namespace vm
}  // namespace oneflow
