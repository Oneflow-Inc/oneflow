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
#include "oneflow/core/vm/vm_desc.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/fuse_phy_instr_operand.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/platform/include/pthread_fork.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include <typeinfo>

namespace oneflow {
namespace vm {

void VirtualMachineEngine::ReleaseInstruction(Instruction* instruction) {
  OF_PROFILER_RANGE_GUARD("R:" + instruction->instr_msg().DebugName());
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
      OF_PROFILER_RANGE_GUARD("E:" + out_instruction->instr_msg().DebugName());
      mut_ready_instruction_list()->PushBack(out_instruction);
    }
  }
}

// Handle pending instructions, and try schedule them to ready list.
void VirtualMachineEngine::HandleLocalPending() {
  OF_PROFILER_RANGE_GUARD("HandleLocalPending");
  InstructionMsgList pending_instr_msgs;
  constexpr static int kPendingHandleWindow = 10;
  GetRewritedPendingInstructionsByWindowSize(kPendingHandleWindow, &pending_instr_msgs);
  InstructionList new_instruction_list;
  INTRUSIVE_FOR_EACH_PTR(instr_msg, &pending_instr_msgs) {
    MakeInstructions(instr_msg, /*out*/ &new_instruction_list);
  }
  INTRUSIVE_FOR_EACH_PTR(instruction, &new_instruction_list) {
    ConsumeMirroredObjects(instruction);
    if (likely(Dispatchable(instruction))) {
      mut_ready_instruction_list()->PushBack(instruction);
      new_instruction_list.Erase(instruction);
    }
  }
}

namespace {

bool FusableBetween(InstructionFuseType fuse_type, InstructionMsg* instr_msg,
                    InstructionMsg* prev_instr_msg) {
  if (unlikely(instr_msg->instr_type_id().instruction_type().fuse_type() != fuse_type)) {
    return false;
  }
  auto* phy_instr_stream = instr_msg->phy_instr_stream();
  if (unlikely(phy_instr_stream == nullptr)) { return false; }
  auto* sequential_dep = instr_msg->phy_instr_operand()->stream_sequential_dependence();
  if (unlikely(sequential_dep == nullptr)) { return false; }

  if (unlikely(prev_instr_msg == nullptr)) { return true; }
  if (unlikely(phy_instr_stream != prev_instr_msg->phy_instr_stream())) { return false; }
  if (unlikely(sequential_dep
               != prev_instr_msg->phy_instr_operand()->stream_sequential_dependence())) {
    return false;
  }
  return true;
}

}  // namespace

void VirtualMachineEngine::MakeAndAppendFusedInstruction(
    InstructionMsgList&& fused_instr_msg_list, InstructionMsgList* /*out*/ pending_instr_msgs) {
  if (unlikely(fused_instr_msg_list.size() == 0)) { return; }
  if (unlikely(fused_instr_msg_list.size() == 1)) {
    fused_instr_msg_list.MoveTo(pending_instr_msgs);
    return;
  }
  auto* begin = fused_instr_msg_list.Begin();
  auto phy_instr_operand = std::make_shared<FusePhyInstrOperand>(std::move(fused_instr_msg_list));
  const auto* stream_tag = begin->phy_instr_stream()->stream_type().stream_tag();
  auto instr_msg = intrusive::make_shared<InstructionMsg>(
      this, std::string(stream_tag) + ".Fuse", begin->phy_instr_parallel_desc(), phy_instr_operand);
  pending_instr_msgs->EmplaceBack(std::move(instr_msg));
}

void VirtualMachineEngine::GetRewritedPendingInstructionsByWindowSize(
    size_t window_size, InstructionMsgList* /*out*/ pending_instr_msgs) {
  InstructionMsgList fused_instr_msg_list;
  INTRUSIVE_FOR_EACH_PTR(instr_msg, mut_local_pending_msg_list()) {
    if (window_size-- <= 0) { break; }
    auto* fuse_begin = fused_instr_msg_list.Begin();
    if (likely(FusableBetween(kEnableInstructionFuseAtAnyPosition, instr_msg, fuse_begin))) {
      // fuse
      mut_local_pending_msg_list()->MoveToDstBack(instr_msg, &fused_instr_msg_list);
    } else if (likely(FusableBetween(kEnableInstructionFuseAsTailOnly, instr_msg, fuse_begin))) {
      // fuse
      mut_local_pending_msg_list()->MoveToDstBack(instr_msg, &fused_instr_msg_list);
      MakeAndAppendFusedInstruction(std::move(fused_instr_msg_list), pending_instr_msgs);
    } else {
      // no fuse
      MakeAndAppendFusedInstruction(std::move(fused_instr_msg_list), pending_instr_msgs);
      mut_local_pending_msg_list()->MoveToDstBack(instr_msg, pending_instr_msgs);
    }
  }
  MakeAndAppendFusedInstruction(std::move(fused_instr_msg_list), pending_instr_msgs);
}

std::string VirtualMachineEngine::GetLivelyInstructionListDebugString(int64_t debug_cnt) {
  std::stringstream ss;
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, mut_lively_instruction_list()) {
    if (--debug_cnt <= 0) { break; }
    ss << instruction->instr_msg().DebugName() << "\n";
  }
  return ss.str();
}

void VirtualMachineEngine::LivelyInstructionListPushBack(Instruction* instruction) {
  ++total_inserted_instruction_cnt_;
  mut_lively_instruction_list()->PushBack(instruction);
}

void VirtualMachineEngine::InsertCallbackProbe(
    const std::function<bool(VirtualMachineEngine*)>& ProbeFunction) {
  callback_probe_list_.EmplaceBack(intrusive::make_shared<CallbackProbe>(ProbeFunction));
}

void VirtualMachineEngine::HandleLocalSchedulerProbe() {
  if (unlikely(local_scheduler_probe_list_.size())) {
    OF_PROFILER_RANGE_PUSH("HandleLocalSchedulerProbe");
    INTRUSIVE_FOR_EACH_PTR(probe, &local_scheduler_probe_list_) {
      probe->probe_function()(this);
      local_scheduler_probe_list_.Erase(probe);
    }
    OF_PROFILER_RANGE_POP();
  }
}

intrusive::shared_ptr<Instruction> VirtualMachineEngine::LivelyInstructionListErase(
    Instruction* instruction, const ScheduleCtx& schedule_ctx) {
  ++total_completed_instruction_cnt_;
  auto ret = mut_lively_instruction_list()->Erase(instruction);
  static constexpr int kFlushInterval = 64;
  if (unlikely(total_completed_instruction_cnt_ % kFlushInterval) == 0) {
    FlushGarbageInstructions(schedule_ctx);
  }
  return ret;
}

// Collect ready instructions onto ready_instruction_list_
void VirtualMachineEngine::ReleaseFinishedInstructions(const ScheduleCtx& schedule_ctx) {
  OF_PROFILER_RANGE_PUSH("ReleaseFinishedInstructions");
  INTRUSIVE_FOR_EACH_PTR(stream, mut_active_stream_list()) {
    while (true) {
      auto* instruction_ptr = stream->mut_running_instruction_list()->Begin();
      if (instruction_ptr == nullptr || !instruction_ptr->Done()) { break; }
      ReleaseInstruction(instruction_ptr);
      stream->mut_running_instruction_list()->Erase(instruction_ptr);
      // By referencing `instruction_ptr->mut_instr_msg()`, we can avoid instr_msg being destructed
      // in stream->DeleteInstruction(...)
      intrusive::shared_ptr<InstructionMsg> instr_msg(instruction_ptr->mut_instr_msg());
      stream->DeleteInstruction(LivelyInstructionListErase(instruction_ptr, schedule_ctx));
      local_garbage_msg_list_.EmplaceBack(std::move(instr_msg));
    }
    if (stream->running_instruction_list().empty()) { mut_active_stream_list()->Erase(stream); }
  }
  OF_PROFILER_RANGE_POP();
}

void VirtualMachineEngine::FlushGarbageInstructions(const ScheduleCtx& schedule_ctx) {
  if (local_garbage_msg_list_.size()) {
    OF_PROFILER_RANGE_PUSH("FlushGarbageInstructions");
    garbage_msg_list_.MoveFrom(&local_garbage_msg_list_);
    schedule_ctx.OnGarbageMsgPending();
    OF_PROFILER_RANGE_POP();
  }
}

int64_t VirtualMachineEngine::this_machine_id() const {
  CHECK_EQ(machine_id_range().size(), 1);
  return machine_id_range().begin();
}

void VirtualMachineEngine::MakeInstructions(InstructionMsg* instr_msg,
                                            /*out*/ InstructionList* new_instruction_list) {
  const auto& instruction_type = instr_msg->instr_type_id().instruction_type();
  bool is_barrier_instruction = instruction_type.IsFrontSequential();
  Stream* stream = CHECK_NOTNULL(instr_msg->phy_instr_stream());
  const auto& pd = instr_msg->phy_instr_parallel_desc();
  intrusive::shared_ptr<Instruction> instr = stream->NewInstruction(instr_msg, pd);
  LivelyInstructionListPushBack(instr.Mutable());
  if (unlikely(is_barrier_instruction)) {
    mut_barrier_instruction_list()->PushBack(instr.Mutable());
  } else {
    new_instruction_list->PushBack(instr.Mutable());
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
  const auto& phy_instr_operand = CHECK_NOTNULL(instruction->instr_msg().phy_instr_operand());
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
void VirtualMachineEngine::DispatchAndPrescheduleInstructions(const ScheduleCtx& schedule_ctx) {
  ReadyInstructionList tmp_ready_instruction_list;
  mut_ready_instruction_list()->MoveTo(&tmp_ready_instruction_list);
  OF_PROFILER_RANGE_GUARD("DispatchAndPrescheduleInstructions");
  INTRUSIVE_FOR_EACH(instruction, &tmp_ready_instruction_list) {
    // Erases `instruction` from tmp_ready_instruction_list before dispatching, because
    // `instruction.dispatched_instruction_hook_` are used in DispatchInstruction.
    tmp_ready_instruction_list.Erase(instruction.Mutable());
    OF_PROFILER_RANGE_GUARD("D:" + instruction->instr_msg().DebugName());
    DispatchInstruction(instruction.Mutable(), schedule_ctx);
    // preschedule instructions
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(edge, instruction->mut_out_edges()) {
      auto* out_instruction = edge->mut_dst_instruction();
      if (Dispatchable(out_instruction)) {
        OF_PROFILER_RANGE_GUARD("P:" + out_instruction->instr_msg().DebugName());
        mut_ready_instruction_list()->PushBack(out_instruction);
      }
    }
  }
}

void VirtualMachineEngine::DispatchInstruction(Instruction* instruction,
                                               const ScheduleCtx& schedule_ctx) {
  auto* stream = instruction->mut_stream();
  stream->mut_running_instruction_list()->PushBack(instruction);
  if (stream->active_stream_hook().empty()) { mut_active_stream_list()->PushBack(stream); }
  const auto& stream_type = stream->stream_type();
  if (OnSchedulerThread(stream_type)) {
    stream_type.Run(instruction);
  } else {
    stream->mut_thread_ctx()->mut_pending_instruction_list()->PushBack(instruction);
    schedule_ctx.OnWorkerLoadPending(stream->mut_thread_ctx());
  }
}

void VirtualMachineEngine::__Init__(const VmDesc& vm_desc) {
  mut_vm_resource_desc()->CopyFrom(vm_desc.vm_resource_desc());
  CHECK_GT(vm_desc.machine_id_range().size(), 0);
  *mut_machine_id_range() = vm_desc.machine_id_range();
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(stream_desc, &vm_desc.stream_type2desc()) {
    if (stream_desc->num_threads() == 0) { continue; }
    auto stream_rt_desc = intrusive::make_shared<StreamRtDesc>(stream_desc);
    mut_stream_type2stream_rt_desc()->Insert(stream_rt_desc.Mutable());
    BalancedSplitter bs(stream_desc->parallel_num(), stream_desc->num_threads());
    for (int64_t i = 0, rel_global_device_id = 0; i < stream_desc->num_threads(); ++i) {
      auto thread_ctx = intrusive::make_shared<ThreadCtx>(stream_rt_desc.Get());
      mut_thread_ctx_list()->PushBack(thread_ctx.Mutable());
      for (int j = bs.At(i).begin(); j < bs.At(i).end(); ++j, ++rel_global_device_id) {
        StreamId stream_id;
        stream_id.__Init__(&stream_desc->stream_type(),
                           this_start_global_device_id() + rel_global_device_id);
        auto stream = intrusive::make_shared<Stream>(
            thread_ctx.Mutable(), stream_id, vm_resource_desc().max_device_num_per_machine());
        stream_rt_desc->add_stream(stream);
        thread_ctx->mut_stream_list()->PushBack(stream.Mutable());
      }
    }
  }
}

void VirtualMachineEngine::GetCachedInstrTypeIdAndPhyInstrStream(const std::string& instr_type_name,
                                                                 int device_id,
                                                                 InstrTypeId* instr_type_id,
                                                                 Stream** stream) {
  auto* cache = &instr_type_name2rt_instr_type_id_;
  auto iter = cache->find(instr_type_name);
  if (unlikely(iter == cache->end())) {
    const auto& instr_type_id_val = LookupInstrTypeId(instr_type_name);
    const auto* stream_type = &instr_type_id_val.stream_type();
    auto* stream_rt_desc = this->mut_stream_type2stream_rt_desc()->FindPtr(stream_type);
    iter = cache->emplace(instr_type_name, RtInstrTypeId(instr_type_id_val, stream_rt_desc)).first;
  }
  instr_type_id->CopyFrom(iter->second.instr_type_id());
  *stream = iter->second.GetStream(device_id);
}

void VirtualMachineEngine::GetInstrTypeIdAndSoleStream(const std::string& instr_type_name,
                                                       InstrTypeId* instr_type_id,
                                                       Stream** stream) {
  instr_type_id->CopyFrom(LookupInstrTypeId(instr_type_name));
  const auto* stream_type = &instr_type_id->stream_type();
  auto* stream_rt_desc = this->mut_stream_type2stream_rt_desc()->FindPtr(stream_type);
  *stream = stream_rt_desc->GetSoleStream();
}

int64_t InstructionMaxRunningSeconds() { return 60 * 5; }

// Returns true if old pending_instruction_list is empty
Maybe<bool> VirtualMachineEngine::Receive(InstructionMsgList* compute_instr_msg_list) {
  OF_PROFILER_RANGE_GUARD("vm:Receive");
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(compute_instr_msg, compute_instr_msg_list) {
    OF_PROFILER_RANGE_PUSH(compute_instr_msg->DebugName());
    OF_PROFILER_RANGE_POP();
  }
  bool old_list_empty = mut_pending_msg_list()->MoveFrom(compute_instr_msg_list);
  return old_list_empty;
}

Maybe<bool> VirtualMachineEngine::Receive(
    intrusive::shared_ptr<InstructionMsg>&& compute_instr_msg) {
  InstructionMsgList instr_msg_list;
  instr_msg_list.EmplaceBack(std::move(compute_instr_msg));
  return Receive(&instr_msg_list);
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
// `lively_instruction_list_.Begin()?->instr_msg().instr_type_id().instruction_type().IsFrontSequential()`
//
void VirtualMachineEngine::TryRunBarrierInstruction(const ScheduleCtx& schedule_ctx) {
  auto* sequnential_instruction = mut_barrier_instruction_list()->Begin();
  CHECK_NOTNULL(sequnential_instruction);
  if (likely(sequnential_instruction != mut_lively_instruction_list()->Begin())) { return; }
  // All instructions before `sequnential_instruction` are handled now, it's time to handle
  // `sequnential_instruction`.
  OF_PROFILER_RANGE_GUARD("RunBarrierInstruction");
  const auto& instr_type_id = sequnential_instruction->instr_msg().instr_type_id();
  const auto& instruction_type = instr_type_id.instruction_type();
  CHECK(instruction_type.IsFrontSequential());
  const StreamType& stream_type = instr_type_id.stream_type();
  CHECK(OnSchedulerThread(stream_type));
  stream_type.Run(sequnential_instruction);
  mut_barrier_instruction_list()->Erase(sequnential_instruction);
  intrusive::shared_ptr<InstructionMsg> instr_msg(sequnential_instruction->mut_instr_msg());
  LivelyInstructionListErase(sequnential_instruction, schedule_ctx);
  local_garbage_msg_list_.EmplaceBack(std::move(instr_msg));
  FlushGarbageInstructions(schedule_ctx);
}

void VirtualMachineEngine::Schedule(const ScheduleCtx& schedule_ctx) {
  // Release finished instructions and try to schedule out instructions in DAG onto ready list.
  if (unlikely(mut_active_stream_list()->size())) { ReleaseFinishedInstructions(schedule_ctx); }
  // Try run the first barrier instruction.
  if (unlikely(mut_barrier_instruction_list()->size())) { TryRunBarrierInstruction(schedule_ctx); }
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
  if (unlikely(mut_ready_instruction_list()->size())) {
    DispatchAndPrescheduleInstructions(schedule_ctx);
  }
  // handle scheduler probes
  if (unlikely(local_scheduler_probe_list_.size())) {
    HandleLocalSchedulerProbe();
  } else if (unlikely(scheduler_probe_list_.thread_unsafe_size())) {
    scheduler_probe_list_.MoveTo(&local_scheduler_probe_list_);
    HandleLocalSchedulerProbe();
  }
}

void VirtualMachineEngine::Callback() {
  InstructionMsgList garbage_msg_list;
  mut_garbage_msg_list()->MoveTo(&garbage_msg_list);
  INTRUSIVE_FOR_EACH(garbage, &garbage_msg_list) {
    CHECK_JUST(Global<ForeignLockHelper>::Get()->WithScopedAcquire([&]() -> Maybe<void> {
      garbage_msg_list.Erase(garbage.Mutable());
      // There may be a tiny gap between appending `garbage` to garbage_list and dereferencing
      // `garbage` in scheduler thread or work thread.
      //  e.g.
      //
      //   void Foo() {
      //     auto garbage = GetGarbage();
      //     AppendToGarbageList(garbage);
      //
      //     // **Callback thread maybe handle garbage in the same time**. From it's point view,
      //     ref_cnt > 1.
      //
      //     garbage.reset(); // explicitly dereference garbage for better understood.
      //   }
      //
      while (garbage->ref_cnt() > 1) {
        // Do nothing. Wait until all other threads ref_cnts released.
      }
      CHECK_NOTNULL(garbage->phy_instr_operand());
      while (garbage->phy_instr_operand().use_count() > 1) {
        // Do nothing. Wait until all other threads ref_cnts released.
      }
      // Destruct garbage.
      return Maybe<void>::Ok();
    }));
    ++total_erased_instruction_cnt_;
  }

  if (unlikely(local_callback_probe_list_.size())) {
    HandleLocalCallbackProbe();
  } else if (unlikely(callback_probe_list_.thread_unsafe_size())) {
    callback_probe_list_.MoveTo(&local_callback_probe_list_);
    HandleLocalCallbackProbe();
  }
}

void VirtualMachineEngine::HandleLocalCallbackProbe() {
  OF_PROFILER_RANGE_PUSH("HandleLocalCallbackProbe");
  INTRUSIVE_FOR_EACH_PTR(probe, &local_callback_probe_list_) {
    if (probe->probe_function()(this)) { local_callback_probe_list_.Erase(probe); }
  }
  OF_PROFILER_RANGE_POP();
}

bool VirtualMachineEngine::SchedulerThreadUnsafeEmpty() const {
  return pending_msg_list().thread_unsafe_size() == 0 && local_pending_msg_list().empty()
         && lively_instruction_list_.empty() && active_stream_list().empty()
         && scheduler_probe_list_.thread_unsafe_size() == 0 && local_scheduler_probe_list_.empty();
}

bool VirtualMachineEngine::SchedulerEmpty() const {
  // hook and size will be check in pending_msg_list().empty().
  return pending_msg_list().empty() && scheduler_probe_list_.empty()
         && SchedulerThreadUnsafeEmpty();
}

bool VirtualMachineEngine::CallbackEmpty() const {
  return (total_erased_instruction_cnt() == total_inserted_instruction_cnt())
         && callback_probe_list_.empty() && local_callback_probe_list_.empty();
}

}  // namespace vm
}  // namespace oneflow
