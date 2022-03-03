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
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/fuse_instruction_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/singleton_ptr.h"

namespace oneflow {
namespace vm {

void VirtualMachineEngine::ReleaseInstruction(Instruction* instruction) {
  OF_PROFILER_RANGE_PUSH("R:" + instruction->DebugName());
  auto* access_list = instruction->mut_access_list();
  INTRUSIVE_FOR_EACH(access, access_list) {
    CHECK_GT(access->ref_cnt(), 1);
    access_list->Erase(access.Mutable());
    auto* mirrored_object = access->mut_mirrored_object();
    if (unlikely(!access->rw_mutexed_object_access_hook().empty())) {
      mirrored_object->mut_access_list()->Erase(access.Mutable());
    }
  }
  auto* out_edges = instruction->mut_out_edges();
  INTRUSIVE_FOR_EACH_PTR(out_edge, out_edges) {
    Instruction* out_instruction = out_edge->mut_dst_instruction();
    // Edges are erased only if the instruction is completed.
    out_edges->Erase(out_edge);
    out_instruction->mut_in_edges()->Erase(out_edge);
    if (Dispatchable(out_instruction)) {
      OF_PROFILER_RANGE_PUSH("E:" + out_instruction->DebugName());
      mut_ready_instruction_list()->PushBack(out_instruction);
      OF_PROFILER_RANGE_POP();
    }
  }
  OF_PROFILER_RANGE_POP();
}

// Handle pending instructions, and try schedule them to ready list.
void VirtualMachineEngine::HandleLocalPending() {
  OF_PROFILER_RANGE_PUSH("HandleLocalPending");
  InstructionList pending_instructions;
  constexpr static int kPendingHandleWindow = 10;
  GetRewritedPendingInstructionsByWindowSize(kPendingHandleWindow, &pending_instructions);
  OF_PROFILER_RANGE_PUSH("InitInstructions");
  InitInstructions(&pending_instructions);
  OF_PROFILER_RANGE_POP();  // "InitInstructions"
  OF_PROFILER_RANGE_PUSH("ConsumeMirroredObjects");
  INTRUSIVE_FOR_EACH_PTR(instruction, &pending_instructions) {
    ConsumeMirroredObjects(instruction);
    if (likely(Dispatchable(instruction))) {
      mut_ready_instruction_list()->PushBack(instruction);
      pending_instructions.Erase(instruction);
    }
  }
  OF_PROFILER_RANGE_POP();  // "ConsumeMirroredObjects"
  OF_PROFILER_RANGE_POP();
}

namespace {

bool FusableBetween(InstructionFuseType fuse_type, Instruction* instruction,
                    Instruction* prev_instruction) {
  if (unlikely(instruction->instruction_type().fuse_type() != fuse_type)) { return false; }
  auto* stream = instruction->mut_stream();
  if (unlikely(stream == nullptr)) { return false; }
  auto* sequential_dep = instruction->phy_instr_operand()->stream_sequential_dependence();
  if (unlikely(sequential_dep == nullptr)) { return false; }

  if (unlikely(prev_instruction == nullptr)) { return true; }
  if (unlikely(stream != prev_instruction->mut_stream())) { return false; }
  if (unlikely(sequential_dep
               != prev_instruction->phy_instr_operand()->stream_sequential_dependence())) {
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
  auto phy_instr_operand = std::make_shared<FusePhyInstrOperand>(std::move(fused_instruction_list));
  auto instruction = intrusive::make_shared<Instruction>(
      begin->mut_stream(), SingletonPtr<FuseInstructionType>(), phy_instr_operand);
  pending_instructions->EmplaceBack(std::move(instruction));
}

void VirtualMachineEngine::GetRewritedPendingInstructionsByWindowSize(
    size_t window_size, InstructionList* /*out*/ pending_instructions) {
  OF_PROFILER_RANGE_PUSH("GetRewritedPendingInstructionsByWindowSize");
  InstructionList fused_instruction_list;
  INTRUSIVE_FOR_EACH_PTR(instruction, mut_local_pending_msg_list()) {
    if (window_size-- <= 0) { break; }
    OF_PROFILER_RANGE_PUSH("Iteration");
    auto* fuse_begin = fused_instruction_list.Begin();
    if (likely(FusableBetween(kEnableInstructionFuseAtAnyPosition, instruction, fuse_begin))) {
      // fuse
      mut_local_pending_msg_list()->MoveToDstBack(instruction, &fused_instruction_list);
    } else if (likely(FusableBetween(kEnableInstructionFuseAsTailOnly, instruction, fuse_begin))) {
      // fuse
      mut_local_pending_msg_list()->MoveToDstBack(instruction, &fused_instruction_list);
      MakeAndAppendFusedInstruction(std::move(fused_instruction_list), pending_instructions);
    } else {
      // no fuse
      MakeAndAppendFusedInstruction(std::move(fused_instruction_list), pending_instructions);
      mut_local_pending_msg_list()->MoveToDstBack(instruction, pending_instructions);
    }
    OF_PROFILER_RANGE_POP();
  }
  MakeAndAppendFusedInstruction(std::move(fused_instruction_list), pending_instructions);
  OF_PROFILER_RANGE_POP();
}

std::string VirtualMachineEngine::GetLivelyInstructionListDebugString(int64_t debug_cnt) {
  std::stringstream ss;
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, mut_lively_instruction_list()) {
    if (--debug_cnt <= 0) { break; }
    ss << instruction->DebugName() << "\n";
  }
  return ss.str();
}

void VirtualMachineEngine::LivelyInstructionListPushBack(Instruction* instruction) {
  ++total_inserted_lively_instruction_cnt_;
  mut_lively_instruction_list()->PushBack(instruction);
}

intrusive::shared_ptr<Instruction> VirtualMachineEngine::LivelyInstructionListErase(
    Instruction* instruction) {
  ++total_erased_lively_instruction_cnt_;
  auto ret = mut_lively_instruction_list()->Erase(instruction);
  static constexpr int kProbeInterval = 20;
  if (unlikely(total_erased_lively_instruction_cnt_ % kProbeInterval) == 0) { HandleProbe(); }
  return ret;
}

void VirtualMachineEngine::InsertProbe(
    const std::function<bool(VirtualMachineEngine*)>& ProbeFunction) {
  probe_list_.EmplaceBack(intrusive::make_shared<Probe>(ProbeFunction));
}

void VirtualMachineEngine::HandleProbe() {
  if (unlikely(probe_list_.thread_unsafe_size())) { probe_list_.MoveTo(&local_probe_list_); }
  HandleLocalProbe();
}

void VirtualMachineEngine::HandleLocalProbe() {
  if (unlikely(local_probe_list_.size())) {
    INTRUSIVE_FOR_EACH_PTR(probe, &local_probe_list_) {
      if (probe->probe(this)) { local_probe_list_.Erase(probe); }
    }
  }
}

// Collect ready instructions onto ready_instruction_list_
void VirtualMachineEngine::ReleaseFinishedInstructions() {
  INTRUSIVE_FOR_EACH_PTR(stream, mut_active_stream_list()) {
    while (true) {
      auto* instruction_ptr = stream->mut_running_instruction_list()->Begin();
      if (instruction_ptr == nullptr || !instruction_ptr->Done()) { break; }
      ReleaseInstruction(instruction_ptr);
      // Prevent destructing instruction_ptr.
      intrusive::shared_ptr<Instruction> instruction =
          stream->mut_running_instruction_list()->Erase(instruction_ptr);
      LivelyInstructionListErase(instruction_ptr);
      instruction_ptr->DeleteStatusAndClearEdges();
      MoveInstructionToGarbageList(std::move(instruction));
    }
    if (stream->running_instruction_list().empty()) { mut_active_stream_list()->Erase(stream); }
  }
}

void VirtualMachineEngine::MoveInstructionToGarbageList(
    intrusive::shared_ptr<Instruction>&& instruction) {
  local_garbage_instruction_list_.EmplaceBack(std::move(instruction));
  static constexpr int kWindowSize = 64;
  // local_garbage_instruction_list_ is the cache of garbage_instruction_list_.
  // `kWindowSize` controls the frequency of the usage of mutexed list.
  if (unlikely(local_garbage_instruction_list_.size() > kWindowSize)) {
    MoveToGarbageListAndNotifyGC();
  }
}

void VirtualMachineEngine::MoveToGarbageListAndNotifyGC() {
  garbage_instruction_list_.MoveFrom(&local_garbage_instruction_list_);
  notify_callback_thread_();
}

void VirtualMachineEngine::InitInstructions(InstructionList* pending_instructions) {
  INTRUSIVE_FOR_EACH_PTR(instruction, pending_instructions) {
    const auto& instruction_type = instruction->instruction_type();
    instruction->InitStatus();
    LivelyInstructionListPushBack(instruction);
    if (unlikely(instruction_type.IsFrontSequential())) {
      pending_instructions->Erase(instruction);
      mut_barrier_instruction_list()->PushBack(instruction);
    }
  }
}

DependenceAccess* VirtualMachineEngine::AccessMirroredObject(OperandAccessType access_type,
                                                             MirroredObject* mirrored_object,
                                                             Instruction* instruction) {
  auto access = access_pool_.make_shared(instruction, mirrored_object, access_type);
  auto* ptr = access.Mutable();
  instruction->mut_access_list()->PushBack(ptr);
  mirrored_object->mut_access_list()->EmplaceBack(std::move(access));
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
  auto* mirrored_object = dst_access->mut_mirrored_object();
  auto* dst_instruction = dst_access->mut_instruction();
  auto* access_list = mirrored_object->mut_access_list();
  if (likely(access_list->Begin() == dst_access)) { return; }
  INTRUSIVE_FOR_EACH_PTR(src_access, access_list) {
    if (unlikely(src_access == dst_access)) { break; }
    TryConnectInstruction(src_access->mut_instruction(), dst_instruction);
    access_list->Erase(src_access);
  }
}

void VirtualMachineEngine::ConnectInstructionsByRead(DependenceAccess* dst_access) {
  CHECK(dst_access->is_const_operand());
  auto* mirrored_object = dst_access->mut_mirrored_object();
  auto* dst_instruction = dst_access->mut_instruction();
  auto* first = mirrored_object->mut_access_list()->Begin();
  if (first->is_mut_operand()) {
    TryConnectInstruction(first->mut_instruction(), dst_instruction);
  } else if (first->is_const_operand()) {
    // do nothing
  } else {
    UNIMPLEMENTED();
  }
}

void VirtualMachineEngine::ConsumeMirroredObjects(Instruction* instruction) {
  const auto& phy_instr_operand = CHECK_NOTNULL(instruction->phy_instr_operand());
  auto* stream_sequential_dep = phy_instr_operand->stream_sequential_dependence();
  if (likely(stream_sequential_dep != nullptr)) {
    ConnectInstructionsByWrite(
        AccessMirroredObject(kMutableOperandAccess, stream_sequential_dep, instruction));
  }
  // Connect instructions by write before connecting by read.
  for (auto* mirrored_object : phy_instr_operand->output_dependences()) {
    ConnectInstructionsByWrite(
        AccessMirroredObject(kMutableOperandAccess, mirrored_object, instruction));
  }
  for (auto* mirrored_object : phy_instr_operand->input_dependences()) {
    ConnectInstructionsByRead(
        AccessMirroredObject(kConstOperandAccess, mirrored_object, instruction));
  }
}

bool VirtualMachineEngine::EdgeDispatchable(const Instruction* src, const Instruction* dst) const {
  return (&src->stream() == &dst->stream()) /* same stream*/
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
void VirtualMachineEngine::DispatchAndPrescheduleInstructions() {
  ReadyInstructionList tmp_ready_instruction_list;
  mut_ready_instruction_list()->MoveTo(&tmp_ready_instruction_list);
  OF_PROFILER_RANGE_PUSH("DispatchAndPrescheduleInstructions");
  INTRUSIVE_FOR_EACH(instruction, &tmp_ready_instruction_list) {
    // Erases `instruction` from tmp_ready_instruction_list before dispatching, because
    // `instruction.dispatched_instruction_hook_` are used in DispatchInstruction.
    tmp_ready_instruction_list.Erase(instruction.Mutable());
    OF_PROFILER_RANGE_PUSH("D:" + instruction->DebugName());
    DispatchInstruction(instruction.Mutable());
    // preschedule instructions
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(edge, instruction->mut_out_edges()) {
      auto* out_instruction = edge->mut_dst_instruction();
      if (Dispatchable(out_instruction)) {
        OF_PROFILER_RANGE_PUSH("P:" + out_instruction->DebugName());
        mut_ready_instruction_list()->PushBack(out_instruction);
        OF_PROFILER_RANGE_POP();
      }
    }
    OF_PROFILER_RANGE_POP();
  }
  OF_PROFILER_RANGE_POP();
}

void VirtualMachineEngine::DispatchInstruction(Instruction* instruction) {
  auto* stream = instruction->mut_stream();
  stream->mut_running_instruction_list()->PushBack(instruction);
  if (stream->active_stream_hook().empty()) { mut_active_stream_list()->PushBack(stream); }
  const auto& stream_type = stream->stream_type();
  if (OnSchedulerThread(stream_type)) {
    stream_type.Run(instruction);
  } else {
    stream->mut_thread_ctx()->mut_worker_pending_instruction_list()->PushBack(instruction);
  }
}

// Returns true if old scheduler_pending_instruction_list is empty
Maybe<bool> VirtualMachineEngine::Receive(InstructionList* compute_instruction_list) {
  OF_PROFILER_RANGE_PUSH("vm:Receive");
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(compute_instruction, compute_instruction_list) {
    OF_PROFILER_RANGE_PUSH(compute_instruction->DebugName());
    OF_PROFILER_RANGE_POP();
  }
  bool old_list_empty = mut_pending_msg_list()->MoveFrom(compute_instruction_list);
  OF_PROFILER_RANGE_POP();
  return old_list_empty;
}

Maybe<bool> VirtualMachineEngine::Receive(
    intrusive::shared_ptr<Instruction>&& compute_instruction) {
  InstructionList instruction_list;
  instruction_list.EmplaceBack(std::move(compute_instruction));
  return Receive(&instruction_list);
}

bool VirtualMachineEngine::OnSchedulerThread(const StreamType& stream_type) {
  return stream_type.OnSchedulerThread() || pthread_fork::IsForkedSubProcess();
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
// `lively_instruction_list_.Begin()?->instruction_type().IsFrontSequential()`
//
void VirtualMachineEngine::TryRunBarrierInstruction() {
  auto* sequnential_instruction = mut_barrier_instruction_list()->Begin();
  CHECK_NOTNULL(sequnential_instruction);
  if (likely(sequnential_instruction != mut_lively_instruction_list()->Begin())) { return; }
  // All instructions before `sequnential_instruction` are handled now, it's time to handle
  // `sequnential_instruction`.
  OF_PROFILER_RANGE_PUSH("RunBarrierInstruction");
  const auto& instruction_type = sequnential_instruction->instruction_type();
  CHECK(instruction_type.IsFrontSequential());
  const StreamType& stream_type = sequnential_instruction->stream().stream_type();
  CHECK(OnSchedulerThread(stream_type));
  stream_type.Run(sequnential_instruction);
  mut_barrier_instruction_list()->Erase(sequnential_instruction);
  LivelyInstructionListErase(sequnential_instruction);
  OF_PROFILER_RANGE_POP();
}

void VirtualMachineEngine::Schedule() {
  // Release finished instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_active_stream_list()->size())) { ReleaseFinishedInstructions(); }
  // Try run the first barrier instruction.
  if (unlikely(mut_barrier_instruction_list()->size())) { TryRunBarrierInstruction(); }
  // Handle pending instructions, and try schedule them to ready list.
  // Use thread_unsafe_size to avoid acquiring mutex lock.
  // The inconsistency between pending_msg_list.list_head_.list_head_.container_ and
  // pending_msg_list.list_head_.list_head_.size_ is not a fatal error because
  // VirtualMachineEngine::Schedule is always in a buzy loop. All instructions will get handled
  // eventually.
  //  VirtualMachineEngine::Receive may be less effiencient if the thread safe version
  //  `pending_msg_list().size()` used here, because VirtualMachineEngine::Schedule is more likely
  //  to get the mutex lock.
  if (unlikely(local_pending_msg_list().size())) {
    HandleLocalPending();
  } else if (unlikely(pending_msg_list().thread_unsafe_size())) {
    // MoveTo is under a lock.
    mut_pending_msg_list()->MoveTo(mut_local_pending_msg_list());
    HandleLocalPending();
  }
  // dispatch ready instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_ready_instruction_list()->size())) { DispatchAndPrescheduleInstructions(); }
  // handle probes
  if (unlikely(local_probe_list_.size())) {
    HandleLocalProbe();
  } else if (unlikely(probe_list_.thread_unsafe_size())) {
    probe_list_.MoveTo(&local_probe_list_);
    HandleLocalProbe();
  }
}

void VirtualMachineEngine::Callback() {
  InstructionList garbage_instruction_list;
  mut_garbage_instruction_list()->MoveTo(&garbage_instruction_list);
  // destruct garbage_instruction_list.
}

void VirtualMachineEngine::NotifyCallback() { MoveToGarbageListAndNotifyGC(); }

bool VirtualMachineEngine::ThreadUnsafeEmpty() const {
  return local_pending_msg_list().empty() && active_stream_list().empty()
         && flying_instruction_cnt() == 0;
}

bool VirtualMachineEngine::Empty() const {
  // hook and size will be check in pending_msg_list().empty().
  return pending_msg_list().empty() && ThreadUnsafeEmpty();
}

bool VirtualMachineEngine::CallbackEmpty() const { return garbage_instruction_list_.empty(); }

}  // namespace vm
}  // namespace oneflow
