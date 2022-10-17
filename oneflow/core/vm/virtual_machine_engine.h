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
#ifndef ONEFLOW_CORE_VM_VIRTUAL_MACHINE_ENGINE_H_
#define ONEFLOW_CORE_VM_VIRTUAL_MACHINE_ENGINE_H_

#include <mutex>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/intrusive/mutexed_list.h"
#include "oneflow/core/intrusive/object_pool.h"
#include "oneflow/core/vm/probe.h"

namespace oneflow {

namespace vm {

class ThreadCtx;

class ScheduleCtx {
 public:
  ScheduleCtx() = default;
  virtual ~ScheduleCtx() = default;

  virtual void OnWorkerLoadPending(vm::ThreadCtx* thread_ctx) const = 0;
};

using ReadyInstructionList =
    intrusive::List<INTRUSIVE_FIELD(Instruction, dispatched_instruction_hook_)>;

class VirtualMachineEngine final : public intrusive::Base {
 public:
  // types
  using ActiveStreamList = intrusive::List<INTRUSIVE_FIELD(Stream, active_stream_hook_)>;
  using ThreadCtxList = intrusive::List<INTRUSIVE_FIELD(ThreadCtx, thread_ctx_hook_)>;
  using InstructionList = intrusive::List<INTRUSIVE_FIELD(Instruction, main_instruction_hook_)>;
  using LivelyInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, lively_instruction_hook_)>;
  using BarrierInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, barrier_instruction_hook_)>;
  using InstructionMutexedList =
      intrusive::MutexedList<INTRUSIVE_FIELD(Instruction, Instruction::main_instruction_hook_)>;

  // Getters
  std::size_t flying_instruction_cnt() const {
    return pending_instruction_list().thread_unsafe_size() + local_pending_instruction_list().size()
           + (total_inserted_instruction_cnt() - total_erased_instruction_cnt());
  }
  size_t total_inserted_instruction_cnt() const { return total_inserted_instruction_cnt_; }
  size_t total_erased_instruction_cnt() const { return total_erased_instruction_cnt_; }
  void InsertProbe(const std::function<bool(VirtualMachineEngine*)>& ProbeFunction);
  const ActiveStreamList& active_stream_list() const { return active_stream_list_; }
  const ThreadCtxList& thread_ctx_list() const { return thread_ctx_list_; }
  const LivelyInstructionList& lively_instruction_list() const { return lively_instruction_list_; }
  const BarrierInstructionList& barrier_instruction_list() const {
    return barrier_instruction_list_;
  }
  const InstructionMutexedList& pending_instruction_list() const {
    return pending_instruction_list_;
  }
  const InstructionList& local_pending_instruction_list() const {
    return local_pending_instruction_list_;
  }
  // Setters
  ActiveStreamList* mut_active_stream_list() { return &active_stream_list_; }
  ThreadCtxList* mut_thread_ctx_list() { return &thread_ctx_list_; }
  LivelyInstructionList* mut_lively_instruction_list() { return &lively_instruction_list_; }
  BarrierInstructionList* mut_barrier_instruction_list() { return &barrier_instruction_list_; }
  InstructionMutexedList* mut_pending_instruction_list() { return &pending_instruction_list_; }
  InstructionList* mut_local_pending_instruction_list() { return &local_pending_instruction_list_; }
  // Returns true if old scheduler_pending_instruction_list is empty
  Maybe<bool> Receive(InstructionList* instr_list);
  void Schedule(const ScheduleCtx& schedule_ctx);
  bool SchedulerThreadUnsafeEmpty() const;
  bool SchedulerEmpty() const;
  std::string GetLivelyInstructionListDebugString(int64_t debug_cnt);
  void MoveToGarbageListAndNotifyGC(const ScheduleCtx& schedule_ctx);

 private:
  ReadyInstructionList* mut_ready_instruction_list() { return &ready_instruction_list_; }

  void ReleaseFinishedInstructions(const ScheduleCtx& schedule_ctx);
  void HandleLocalPending();
  void FetchAndTryFusePendingInstructions(InstructionList* /*out*/ pending_instructions);
  void MakeAndAppendFusedInstruction(InstructionList&& fused_instruction_list,
                                     InstructionList* /*out*/ pending_instructions);
  void TryRunBarrierInstruction(const ScheduleCtx& schedule_ctx);
  void DispatchAndPrescheduleInstructions(const ScheduleCtx& schedule_ctx);
  bool OnSchedulerThread(const vm::Stream& stream);

  void ReleaseInstruction(Instruction* instruction);

  void TryConnectInstruction(Instruction* src_instruction, Instruction* dst_instruction);
  void ConnectInstructionsByWrite(DependenceAccess* dst_access);
  void ConnectInstructionsByRead(DependenceAccess* dst_access);
  DependenceAccess* AccessDependence(OperandAccessType access_type, Dependence* dependence,
                                     Instruction* instrution);
  void ConsumeDependences(Instruction* instruction);
  void DispatchInstruction(Instruction* instruction, const ScheduleCtx& schedule_ctx);

  bool EdgeDispatchable(const Instruction* src, const Instruction* dst) const;
  bool Dispatchable(Instruction* instruction) const;

  void TryDispatchReadyInstructions();

  void LivelyInstructionListPushBack(Instruction* instruction);
  intrusive::shared_ptr<Instruction> LivelyInstructionListErase(Instruction* instruction);
  void HandleLocalProbe();

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  VirtualMachineEngine()
      : intrusive_ref_(),
        active_stream_list_(),
        thread_ctx_list_(),
        pending_instruction_mutex_(),
        pending_instruction_list_(&pending_instruction_mutex_),
        local_pending_instruction_list_(),
        ready_instruction_list_(),
        lively_instruction_list_(),
        total_inserted_instruction_cnt_(0),
        total_erased_instruction_cnt_(0),
        probe_mutex_(),
        probe_list_(&probe_mutex_),
        local_probe_list_(),
        barrier_instruction_list_() {}
  intrusive::Ref intrusive_ref_;
  // lists or maps
  // Do not change the order of the following fields
  ActiveStreamList active_stream_list_;
  ThreadCtxList thread_ctx_list_;
  std::mutex pending_instruction_mutex_;
  InstructionMutexedList pending_instruction_list_;
  // local_pending_instruction_list_ should be consider as the cache of pending_instruction_list_.
  InstructionList local_pending_instruction_list_;
  ReadyInstructionList ready_instruction_list_;
  LivelyInstructionList lively_instruction_list_;
  size_t total_inserted_instruction_cnt_;
  size_t total_erased_instruction_cnt_;

  using VmProbe = Probe<std::function<bool(VirtualMachineEngine*)>>;
  std::mutex probe_mutex_;
  intrusive::MutexedList<INTRUSIVE_FIELD(VmProbe, probe_hook_)> probe_list_;
  intrusive::List<INTRUSIVE_FIELD(VmProbe, probe_hook_)> local_probe_list_;

  BarrierInstructionList barrier_instruction_list_;
  DependenceAccess::object_pool_type access_pool_;
  InstructionEdge::object_pool_type instruction_edge_pool_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VIRTUAL_MACHINE_ENGINE_H_
