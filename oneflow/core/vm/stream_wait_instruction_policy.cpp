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
#include "oneflow/core/vm/stream_wait_instruction_policy.h"
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

StreamWaitInstructionPolicy::StreamWaitInstructionPolicy(
    small_vector<intrusive::shared_ptr<LocalDepObject>>&& dependences, vm::Stream* from_vm_stream,
    vm::Stream* to_vm_stream)
    : dependences_(std::move(dependences)),
      input_dependences_(),
      output_dependences_(),
      from_vm_stream_(from_vm_stream) {
  for (const auto& dep : dependences_) { output_dependences_.push_back(dep.get()); }
  stream_sequential_dependence_ = to_vm_stream->schedule_local_dep_object().get();
}

bool StreamWaitInstructionPolicy::Prescheduleable(const Stream* src, const Stream* dst) const {
  return &src->thread_ctx() == &dst->thread_ctx();
}

void StreamWaitInstructionPolicy::InitInstructionStatus(Instruction* instruction) {
  auto* stream = instruction->mut_stream();
  auto* ep_stream_policy_base =
      CHECK_NOTNULL(dynamic_cast<EpStreamPolicyBase*>(instruction->mut_stream_policy()));
  ep_stream_policy_base->InitInstructionStatus(*stream, instruction->mut_status_buffer());
  auto* ep_event_provider = ep_stream_policy_base->ep_event_provider();
  const auto& ep_event = CHECK_NOTNULL(ep_event_provider)->GetReusedEpEvent();
  mut_ep_event() = ep_event;
}

void StreamWaitInstructionPolicy::DeleteInstructionStatus(Instruction* instruction) {
  auto* stream = instruction->mut_stream();
  instruction->stream_policy().DeleteInstructionStatus(*stream, instruction->mut_status_buffer());
  mut_ep_event().reset();
}

void StreamWaitInstructionPolicy::Compute(vm::Instruction* instruction) {
  const auto& ep_event = mut_ep_event();
  {
    // Record event.
    auto* from_naive_stream_policy =
        dynamic_cast<EpStreamPolicyBase*>(mut_from_vm_stream()->mut_stream_policy());
    CHECK_NOTNULL(from_naive_stream_policy);
    auto* from_stream = from_naive_stream_policy->stream();
    from_stream->RecordEvent(ep_event->mut_event());
  }
  {
    // Wait event.
    auto* to_ep_stream_policy_base =
        dynamic_cast<EpStreamPolicyBase*>(instruction->mut_stream()->mut_stream_policy());
    CHECK_NOTNULL(to_ep_stream_policy_base);
    auto* to_ep_stream = to_ep_stream_policy_base->stream();
    CHECK_EQ(ep_event->mut_device(), to_ep_stream->device())
        << "only support waiting events from same device";
    ep_event->mut_device()->SetAsActiveDevice();
#ifdef WITH_CUDA

    auto* ep_cuda_event = CHECK_NOTNULL(dynamic_cast<ep::CudaEvent*>(ep_event->mut_event()));
    auto* ep_cuda_stream = CHECK_NOTNULL(dynamic_cast<ep::CudaStream*>(to_ep_stream));

    OF_CUDA_CHECK(
        cudaStreamWaitEvent(ep_cuda_stream->cuda_stream(), ep_cuda_event->cuda_event(), 0));
#else
    UNIMPLEMENTED();
#endif  // WITH_CUDA
  }
}

}  // namespace vm
}  // namespace oneflow
