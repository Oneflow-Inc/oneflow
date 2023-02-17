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
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/common/env_var/vm.h"
#include "oneflow/core/vm/caching_allocator.h"
#include "oneflow/core/vm/fuse_instruction_policy.h"
#include "oneflow/core/vm/release_tensor_instruction_policy.h"
#include "oneflow/core/vm/allocator.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/extension/stack/foreign_stack_getter.h"

namespace oneflow {

namespace vm {

void VirtualMachineEngine::ReleaseInstruction(Instruction* instruction) {
  OF_PROFILER_RANGE_GUARD("R:" + instruction->DebugName());
  auto* access_list = instruction->mut_access_list();
  INTRUSIVE_FOR_EACH(access, access_list) {
    CHECK_GT(access->ref_cnt(), 1);
    access_list->Erase(access.Mutable());
    auto* dependence = access->mut_dependence();
    if (unlikely(!access->rw_mutexed_object_access_hook().empty())) {
      dependence->mut_access_list()->Erase(access.Mutable());
    }
  }
  auto* out_edges = instruction->mut_out_edges();
  INTRUSIVE_FOR_EACH_PTR(out_edge, out_edges) {
    Instruction* out_instruction = out_edge->mut_dst_instruction();
    // Edges are erased only if the instruction is completed.
    out_edges->Erase(out_edge);
    out_instruction->mut_in_edges()->Erase(out_edge);
    if (Dispatchable(out_instruction)) {
      OF_PROFILER_RANGE_GUARD("E:" + out_instruction->DebugName());
      mut_ready_instruction_list()->PushBack(out_instruction);
    }
  }
}

// Handle pending instructions, and try schedule them to ready list.
void VirtualMachineEngine::HandleLocalPending() {
  OF_PROFILER_RANGE_GUARD("HandleLocalPending");
  InstructionList pending_instructions;
  FetchAndTryFusePendingInstructions(&pending_instructions);
  INTRUSIVE_FOR_EACH_PTR(instruction, &pending_instructions) {
    const auto& instruction_policy = instruction->instruction_policy();
    instruction->InitStatus();
    LivelyInstructionListPushBack(instruction);
    if (unlikely(instruction_policy.IsBarrier())) {
      mut_barrier_instruction_list()->PushBack(instruction);
    } else {
      ConsumeDependences(instruction);
      if (likely(Dispatchable(instruction))) {
        mut_ready_instruction_list()->PushBack(instruction);
      }
    }
  }
}

namespace {

bool FusableBetween(InstructionFuseType fuse_type, Instruction* instruction,
                    Instruction* prev_instruction) {
  if (unlikely(instruction->instruction_policy().fuse_type() != fuse_type)) { return false; }
  auto* stream = instruction->mut_stream();
  if (unlikely(stream == nullptr)) { return false; }
  auto* sequential_dep = instruction->instruction_policy().stream_sequential_dependence();
  if (unlikely(sequential_dep == nullptr)) { return false; }

  if (unlikely(prev_instruction == nullptr)) { return true; }
  if (unlikely(stream != prev_instruction->mut_stream())) { return false; }
  if (unlikely(sequential_dep
               != prev_instruction->instruction_policy().stream_sequential_dependence())) {
    return false;
  }
  return true;
}

}  // namespace

void VirtualMachineEngine::MakeAndAppendFusedInstruction(
    InstructionList&& fused_instruction_list, InstructionList* /*out*/ pending_instructions) {
  if (unlikely(fused_instruction_list.size() == 0)) { return; }
  if (unlikely(fused_instruction_list.size() == 1)) {
    fused_instruction_list.MoveTo(pending_instructions);
    return;
  }
  auto* begin = fused_instruction_list.Begin();
  auto instruction = intrusive::make_shared<Instruction>(
      begin->mut_stream(),
      std::make_shared<FuseInstructionPolicy>(std::move(fused_instruction_list)));
  pending_instructions->EmplaceBack(std::move(instruction));
}

void VirtualMachineEngine::FetchAndTryFusePendingInstructions(
    InstructionList* /*out*/ pending_instructions) {
  size_t window_size = ThreadLocalEnvInteger<ONEFLOW_VM_PENDING_HANDLE_WINDOW_SIZE>();
  InstructionList fused_instruction_list;
  INTRUSIVE_FOR_EACH_PTR(instruction, mut_local_pending_instruction_list()) {
    if (window_size-- <= 0) { break; }
    auto* fuse_begin = fused_instruction_list.Begin();
    if (likely(FusableBetween(kEnableInstructionFuseAtAnyPosition, instruction, fuse_begin))) {
      // fuse
      mut_local_pending_instruction_list()->MoveToDstBack(instruction, &fused_instruction_list);
    } else if (likely(FusableBetween(kEnableInstructionFuseAsTailOnly, instruction, fuse_begin))) {
      // fuse
      mut_local_pending_instruction_list()->MoveToDstBack(instruction, &fused_instruction_list);
      MakeAndAppendFusedInstruction(std::move(fused_instruction_list), pending_instructions);
    } else {
      // no fuse
      MakeAndAppendFusedInstruction(std::move(fused_instruction_list), pending_instructions);
      mut_local_pending_instruction_list()->MoveToDstBack(instruction, pending_instructions);
    }
  }
  MakeAndAppendFusedInstruction(std::move(fused_instruction_list), pending_instructions);
}

std::string VirtualMachineEngine::GetLivelyInstructionListDebugString(int64_t debug_cnt) {
  std::stringstream ss;
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, mut_lively_instruction_list()) {
    if (--debug_cnt <= 0) { break; }
    ss << instruction->DebugName() << " ptr: " << instruction
       << " dispatched:" << (instruction->dispatched_instruction_hook().empty() ? "0" : "1")
       << " launched:" << (instruction->Launched() ? "1" : "0")
       << " done:" << (instruction->Done() ? "1" : "0");
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(edge, instruction->mut_in_edges()) {
      ss << " dep-ptr:" << &edge->src_instruction();
    }
    ss << "\n";
  }
  return ss.str();
}

void VirtualMachineEngine::LivelyInstructionListPushBack(Instruction* instruction) {
  ++total_inserted_instruction_cnt_;
  mut_lively_instruction_list()->PushBack(instruction);
}

void VirtualMachineEngine::InsertProbe(
    const std::function<bool(VirtualMachineEngine*)>& ProbeFunction) {
  probe_list_.EmplaceBack(intrusive::make_shared<VmProbe>(ProbeFunction));
}

void VirtualMachineEngine::HandleLocalProbe() {
  OF_PROFILER_RANGE_GUARD("HandleLocalProbe");
  if (unlikely(local_probe_list_.size())) {
    OF_PROFILER_RANGE_PUSH("HandleLocalProbe");
    INTRUSIVE_FOR_EACH_PTR(probe, &local_probe_list_) {
      if (probe->probe_function()(this)) { local_probe_list_.Erase(probe); }
    }
    OF_PROFILER_RANGE_POP();
  }
}

intrusive::shared_ptr<Instruction> VirtualMachineEngine::LivelyInstructionListErase(
    Instruction* instruction) {
  ++total_erased_instruction_cnt_;
  return mut_lively_instruction_list()->Erase(instruction);
}

// Collect ready instructions onto ready_instruction_list_
void VirtualMachineEngine::ReleaseFinishedInstructions(const ScheduleCtx& schedule_ctx) {
  INTRUSIVE_FOR_EACH_PTR(stream, mut_active_stream_list()) {
    while (true) {
      auto* instruction_ptr = stream->mut_running_instruction_list()->Begin();
      if (instruction_ptr == nullptr) { break; }
      if (!(instruction_ptr->in_edges().empty() && instruction_ptr->Done())) { break; }
      ReleaseInstruction(instruction_ptr);
      // Prevent destructing instruction_ptr.
      intrusive::shared_ptr<Instruction> instruction =
          stream->mut_running_instruction_list()->Erase(instruction_ptr);
      LivelyInstructionListErase(instruction_ptr);
      instruction_ptr->DeleteStatusAndCheckEdges();
    }
    if (stream->running_instruction_list().empty()) { mut_active_stream_list()->Erase(stream); }
  }
}

DependenceAccess* VirtualMachineEngine::AccessDependence(OperandAccessType access_type,
                                                         Dependence* dependence,
                                                         Instruction* instruction) {
  auto access = access_pool_.make_shared(instruction, dependence, access_type);
  auto* ptr = access.Mutable();
  instruction->mut_access_list()->PushBack(ptr);
  dependence->mut_access_list()->EmplaceBack(std::move(access));
  return ptr;
}

void VirtualMachineEngine::TryConnectInstruction(Instruction* src_instruction,
                                                 Instruction* dst_instruction) {
  if (unlikely(src_instruction == dst_instruction)) { return; }
  if (likely(EdgeDispatchable(src_instruction, dst_instruction))) { return; }
  auto edge = instruction_edge_pool_.make_shared(src_instruction, dst_instruction);
  src_instruction->mut_out_edges()->PushBack(edge.Mutable());
  dst_instruction->mut_in_edges()->PushBack(edge.Mutable());
}

void VirtualMachineEngine::ConnectInstructionsByWrite(DependenceAccess* dst_access) {
  CHECK(dst_access->is_mut_operand());
  auto* dependence = dst_access->mut_dependence();
  auto* dst_instruction = dst_access->mut_instruction();
  auto* access_list = dependence->mut_access_list();
  if (likely(access_list->Begin() == dst_access)) { return; }
  INTRUSIVE_FOR_EACH_PTR(src_access, access_list) {
    if (unlikely(src_access == dst_access)) { break; }
    TryConnectInstruction(src_access->mut_instruction(), dst_instruction);
    access_list->Erase(src_access);
  }
}

void VirtualMachineEngine::ConnectInstructionsByRead(DependenceAccess* dst_access) {
  CHECK(dst_access->is_const_operand());
  auto* dependence = dst_access->mut_dependence();
  auto* dst_instruction = dst_access->mut_instruction();
  auto* first = dependence->mut_access_list()->Begin();
  if (first->is_mut_operand()) {
    TryConnectInstruction(first->mut_instruction(), dst_instruction);
  } else if (first->is_const_operand()) {
    // do nothing
  } else {
    UNIMPLEMENTED();
  }
}

void VirtualMachineEngine::ConsumeDependences(Instruction* instruction) {
  const auto& instruction_policy = instruction->instruction_policy();
  auto* stream_sequential_dep = instruction_policy.stream_sequential_dependence();
  if (likely(stream_sequential_dep != nullptr)) {
    ConnectInstructionsByWrite(
        AccessDependence(kMutableOperandAccess, stream_sequential_dep, instruction));
  }
  // Connect instructions by write before connecting by read.
  for (auto* dependence : instruction_policy.output_dependences()) {
    ConnectInstructionsByWrite(AccessDependence(kMutableOperandAccess, dependence, instruction));
  }
  for (auto* dependence : instruction_policy.input_dependences()) {
    ConnectInstructionsByRead(AccessDependence(kConstOperandAccess, dependence, instruction));
  }
}

bool VirtualMachineEngine::EdgeDispatchable(const Instruction* src, const Instruction* dst) const {
  return dst->instruction_policy().Prescheduleable(&src->stream(), &dst->stream())
         && !src->dispatched_instruction_hook().empty() /* dispatched */;
}

bool VirtualMachineEngine::Dispatchable(Instruction* instruction) const {
  if (unlikely(!instruction->dispatched_instruction_hook().empty())) { return false; }
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(edge, instruction->mut_in_edges()) {
    const auto* src_instruction = &edge->src_instruction();
    if (unlikely(!EdgeDispatchable(src_instruction, instruction))) { return false; }
  }
  return true;
}

// Dispatch ready instructions and put prescheduled instructions onto ready_instruction_list_.
void VirtualMachineEngine::DispatchAndPrescheduleInstructions(const ScheduleCtx& schedule_ctx) {
  OF_PROFILER_RANGE_GUARD("DispatchAndPrescheduleInstructions");
  ReadyInstructionList tmp_ready_instruction_list;
  mut_ready_instruction_list()->MoveTo(&tmp_ready_instruction_list);
  INTRUSIVE_FOR_EACH(instruction, &tmp_ready_instruction_list) {
    // Erases `instruction` from tmp_ready_instruction_list before dispatching, because
    // `instruction.dispatched_instruction_hook_` are used in DispatchInstruction.
    tmp_ready_instruction_list.Erase(instruction.Mutable());
    OF_PROFILER_RANGE_GUARD("D:" + instruction->DebugName());
    DispatchInstruction(instruction.Mutable(), schedule_ctx);
    // preschedule instructions
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(edge, instruction->mut_out_edges()) {
      auto* out_instruction = edge->mut_dst_instruction();
      if (Dispatchable(out_instruction)) {
        OF_PROFILER_RANGE_GUARD("P:" + out_instruction->DebugName());
        mut_ready_instruction_list()->PushBack(out_instruction);
      }
    }
  }
}

namespace {

std::string DebugDeviceReset(vm::Stream* stream) {
  stream->mut_stream_policy()->mut_allocator()->DeviceReset();
  return "reset device";
}

}  // namespace

void VirtualMachineEngine::DispatchInstruction(Instruction* instruction,
                                               const ScheduleCtx& schedule_ctx) {
  ForeignFrameThreadLocalGuard guard(instruction->foreign_frame());
  auto* stream = instruction->mut_stream();
  // Prepare
  {
    const auto& ret = TRY(instruction->Prepare());
    if (unlikely(!ret.IsOk())) {
      if (ret.error()->has_out_of_memory_error()) {
        CHECK_JUST_MSG(ret, std::stringstream() << DebugDeviceReset(stream));
      } else {
        CHECK_JUST(ret);
      }
    }
  }
  stream->mut_running_instruction_list()->PushBack(instruction);
  if (stream->active_stream_hook().empty()) { mut_active_stream_list()->PushBack(stream); }
  // Compute
  if (OnSchedulerThread(*stream)) {
    stream->stream_policy().RunIf(instruction);
  } else {
    stream->mut_thread_ctx()->mut_worker_pending_instruction_list()->PushBack(instruction);
    schedule_ctx.OnWorkerLoadPending(stream->mut_thread_ctx());
  }
}

// Returns true if old scheduler_pending_instruction_list is empty
Maybe<bool> VirtualMachineEngine::Receive(InstructionList* compute_instruction_list) {
  OF_PROFILER_RANGE_GUARD("vm:Receive");
#ifdef OF_ENABLE_PROFILER
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(compute_instruction, compute_instruction_list) {
    OF_PROFILER_RANGE_GUARD(compute_instruction->DebugName());
  }
#endif

  bool old_list_empty = mut_pending_instruction_list()->MoveFrom(compute_instruction_list);
  return old_list_empty;
}

bool VirtualMachineEngine::OnSchedulerThread(const Stream& stream) {
  return stream.on_scheduler_thread() || pthread_fork::IsForkedSubProcess();
}

// Barrier instructions are run after all previous lively instructions.
//
// `instruction.lively_instruction_hook_` is linked to `vm.lively_instruction_list_` for all
// instructions. `instruction.barrier_instruction_list_` is linked to `vm.barrier_instruction_list_`
// only for barrier instructions.
//
//
//  e.g. case0: waiting other instructions done.
//
//  +---------------------------+   +---------------------------+   +---------------------------+
//  |      virtual_machine      |   |        instruction0       |   |        instruction1       |
//  +---------------------------+   +---------------------------+   +---------------------------+
//  |            ...            |   |            ...            |   |            ...            |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  | lively_instruction_list_  |<->| lively_instruction_hook_  |<->| lively_instruction_hook_  |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  |            ...            |   |            ...            |   |            ...            |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  | barrier_instruction_list_ |<+ | barrier_instruction_hook_ | +>| barrier_instruction_hook_ |
//  |---------------------------| | |---------------------------| | |---------------------------|
//  |            ...            | | |            ...            | | |            ...            |
//  +---------------------------+ | +---------------------------+ | +---------------------------+
//                                |                               |
//                                +-------------------------------+
//
// `instruction1` is a barrier instruction with barrier_instruction_hook_ linked, while
// instruction0 is not. From the `virtual_machine`'s view, `barrier_instruction_list_.Begin() !=
// lively_instruction_list_.Begin()`, so it's not the time to run barrier instruction
// `barrier_instruction_list_.Begin()`.
//
//
//  e.g. case1: run barrier instructions.
//
//  +---------------------------+   +---------------------------+   +---------------------------+
//  |      virtual_machine      |   |        instruction0       |   |        instruction1       |
//  +---------------------------+   +---------------------------+   +---------------------------+
//  |            ...            |   |            ...            |   |            ...            |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  | lively_instruction_list_  |<->| lively_instruction_hook_  |<->| lively_instruction_hook_  |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  |            ...            |   |            ...            |   |            ...            |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  | barrier_instruction_list_ |<->| barrier_instruction_hook_ |   | barrier_instruction_hook_ |
//  |---------------------------|   |---------------------------|   |---------------------------|
//  |            ...            |   |            ...            |   |            ...            |
//  +---------------------------+   +---------------------------+   +---------------------------+
//
// `instruction0` is a barrier instruction with barrier_instruction_hook_ linked.
// From the `virtual_machine`'s view, `barrier_instruction_list_.Begin() ==
// lively_instruction_list_.Begin()`, so it's the time to run barrier instruction
// `barrier_instruction_list_.Begin()`.
//
//
// With the introduction of barrier_instruction_list_/barrier_instruction_hook_, the function
// VirtualMachineEngine::Schedule can achive higher performance. For the most cases, barrier
// instructions are scarcely received by vm, there is no need for vm to run
// VirtualMachineEngine::TryRunBarrierInstruction every time VirtualMachineEngine::Schedule run. On
// the other hand, `barrier_instruction_hook_.size() == 0` is more lightweight than
// `lively_instruction_list_.Begin()?->instruction_policy().IsBarrier()`
//
void VirtualMachineEngine::TryRunBarrierInstruction(const ScheduleCtx& schedule_ctx) {
  auto* sequnential_instruction = mut_barrier_instruction_list()->Begin();
  CHECK_NOTNULL(sequnential_instruction);
  if (likely(sequnential_instruction != mut_lively_instruction_list()->Begin())) { return; }
  // All instructions before `sequnential_instruction` are handled now, it's time to handle
  // `sequnential_instruction`.
  OF_PROFILER_RANGE_GUARD("TryRunBarrierInstruction");
  const auto& instruction_policy = sequnential_instruction->instruction_policy();
  CHECK(instruction_policy.IsBarrier());
  CHECK(OnSchedulerThread(sequnential_instruction->stream()));
  const StreamPolicy& stream_policy = sequnential_instruction->stream().stream_policy();
  stream_policy.RunIf(sequnential_instruction);
  mut_barrier_instruction_list()->Erase(sequnential_instruction);
  LivelyInstructionListErase(sequnential_instruction);
}

void VirtualMachineEngine::Schedule(const ScheduleCtx& schedule_ctx) {
  // Release finished instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_active_stream_list()->size())) { ReleaseFinishedInstructions(schedule_ctx); }
  // Try run the first barrier instruction.
  if (unlikely(mut_barrier_instruction_list()->size())) { TryRunBarrierInstruction(schedule_ctx); }
  // Handle pending instructions, and try schedule them to ready list.
  // Use thread_unsafe_size to avoid acquiring mutex lock.
  // The inconsistency between pending_instruction_list.list_head_.list_head_.container_ and
  // pending_instruction_list.list_head_.list_head_.size_ is not a fatal error because
  // VirtualMachineEngine::Schedule is always in a busy loop. All instructions will get handled
  // eventually.
  //  VirtualMachineEngine::Receive may be less effiencient if the thread safe version
  //  `pending_instruction_list().size()` used here, because VirtualMachineEngine::Schedule is more
  //  likely to get the mutex lock.
  if (unlikely(local_pending_instruction_list().size())) {
    HandleLocalPending();
  } else if (unlikely(pending_instruction_list().thread_unsafe_size())) {
    // MoveTo is under a lock.
    mut_pending_instruction_list()->MoveTo(mut_local_pending_instruction_list());
    if (local_pending_instruction_list().size()) { HandleLocalPending(); }
  }
  // dispatch ready instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_ready_instruction_list()->size())) {
    DispatchAndPrescheduleInstructions(schedule_ctx);
  }
  // handle scheduler probes
  if (unlikely(local_probe_list_.size())) {
    HandleLocalProbe();
  } else if (unlikely(probe_list_.thread_unsafe_size())) {
    probe_list_.MoveTo(&local_probe_list_);
    if (local_probe_list_.size()) { HandleLocalProbe(); }
  }
}

bool VirtualMachineEngine::SchedulerThreadUnsafeEmpty() const {
  return pending_instruction_list().thread_unsafe_size() == 0
         && local_pending_instruction_list().empty() && lively_instruction_list_.empty()
         && active_stream_list().empty() && probe_list_.thread_unsafe_size() == 0
         && local_probe_list_.empty();
}

bool VirtualMachineEngine::SchedulerEmpty() const {
  // hook and size will be check in pending_instruction_list().empty().
  return pending_instruction_list().empty() && probe_list_.empty() && SchedulerThreadUnsafeEmpty();
}

}  // namespace vm
}  // namespace oneflow
