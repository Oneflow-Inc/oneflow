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
#include "oneflow/core/vm/stream_wait_event_instruction_policy.h"
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

StreamWaitEventInstructionPolicy::StreamWaitEventInstructionPolicy(
    const small_vector<intrusive::shared_ptr<LocalDepObject>>& dependences,
    const std::shared_ptr<StreamRecordEventInstructionPolicy>&
        stream_record_event_instruction_policy)
    : dependences_(dependences),
      input_dependences_(),
      output_dependences_(),
      stream_record_event_instruction_policy_(stream_record_event_instruction_policy) {
  for (const auto& dep : dependences_) { output_dependences_.push_back(dep.get()); }
}

void StreamWaitEventInstructionPolicy::DeleteInstructionStatus(Instruction* instruction) {
  auto* stream = instruction->mut_stream();
  instruction->stream_policy().DeleteInstructionStatus(*stream, instruction->mut_status_buffer());
  stream_record_event_instruction_policy_->mut_ep_event().reset();
}

void StreamWaitEventInstructionPolicy::Compute(vm::Instruction* instruction) {
  const auto& ep_event = stream_record_event_instruction_policy_->mut_ep_event();
  // Wait event.
  auto* ep_stream_policy_base =
      dynamic_cast<EpStreamPolicyBase*>(instruction->mut_stream()->mut_stream_policy());
  CHECK_NOTNULL(ep_stream_policy_base);
  auto* ep_stream = ep_stream_policy_base->stream();
  CHECK_EQ(ep_event->mut_device(), ep_stream->device())
      << "only support waiting events from same device";
  ep_event->mut_device()->SetAsActiveDevice();
#ifdef WITH_CUDA

  auto* ep_cuda_event = CHECK_NOTNULL(dynamic_cast<ep::CudaEvent*>(ep_event->mut_event()));
  auto* ep_cuda_stream = CHECK_NOTNULL(dynamic_cast<ep::CudaStream*>(ep_stream));

  OF_CUDA_CHECK(cudaStreamWaitEvent(ep_cuda_stream->cuda_stream(), ep_cuda_event->cuda_event(), 0));
#else
  UNIMPLEMENTED();
#endif  // WITH_CUDA
}

}  // namespace vm
}  // namespace oneflow
