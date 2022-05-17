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
#include "oneflow/core/vm/stream_runtime_desc.h"
#include "oneflow/core/vm/runtime_instr_type_id.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/vm/vm_resource_desc.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/job/parallel_desc.h"
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

  virtual void OnGarbageMsgPending() const = 0;
  virtual void OnWorkerLoadPending(vm::ThreadCtx* thread_ctx) const = 0;
};

class VmDesc;
class VirtualMachineEngine final : public intrusive::Base {
 public:
  // types
  using ActiveStreamList = intrusive::List<INTRUSIVE_FIELD(Stream, active_stream_hook_)>;
  using ThreadCtxList = intrusive::List<INTRUSIVE_FIELD(ThreadCtx, thread_ctx_hook_)>;
  using InstructionList = intrusive::List<INTRUSIVE_FIELD(Instruction, instruction_hook_)>;
  using LivelyInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, lively_instruction_hook_)>;
  using BarrierInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, barrier_instruction_hook_)>;
  using InstructionMsgMutexedList =
      intrusive::MutexedList<INTRUSIVE_FIELD(InstructionMsg, InstructionMsg::instr_msg_hook_)>;
  using StreamType2StreamRtDesc =
      intrusive::SkipList<INTRUSIVE_FIELD(StreamRtDesc, stream_type_key_)>;

  // Getters
  const VmResourceDesc& vm_resource_desc() const {
    if (vm_resource_desc_) { return vm_resource_desc_.Get(); }
    static const auto default_val = intrusive::make_shared<VmResourceDesc>();
    return default_val.Get();
  }
  const Range& machine_id_range() const { return machine_id_range_; }
  std::size_t flying_instruction_cnt() const {
    return pending_msg_list().thread_unsafe_size() + local_pending_msg_list().size()
           + (total_inserted_instruction_cnt() - total_erased_instruction_cnt());
  }
  size_t total_inserted_instruction_cnt() const { return total_inserted_instruction_cnt_; }
  size_t total_erased_instruction_cnt() const { return total_erased_instruction_cnt_; }
  void InsertCallbackProbe(const std::function<bool(VirtualMachineEngine*)>& ProbeFunction);
  const ActiveStreamList& active_stream_list() const { return active_stream_list_; }
  const ThreadCtxList& thread_ctx_list() const { return thread_ctx_list_; }
  const LivelyInstructionList& lively_instruction_list() const { return lively_instruction_list_; }
  const BarrierInstructionList& barrier_instruction_list() const {
    return barrier_instruction_list_;
  }
  const InstructionMsgMutexedList& pending_msg_list() const { return pending_msg_list_; }
  const InstructionMsgList& local_pending_msg_list() const { return local_pending_msg_list_; }
  const StreamType2StreamRtDesc& stream_type2stream_rt_desc() const {
    return stream_type2stream_rt_desc_;
  }
  // Setters
  VmResourceDesc* mut_vm_resource_desc() {
    if (!vm_resource_desc_) { vm_resource_desc_ = intrusive::make_shared<VmResourceDesc>(); }
    return vm_resource_desc_.Mutable();
  }
  Range* mut_machine_id_range() { return &machine_id_range_; }
  ActiveStreamList* mut_active_stream_list() { return &active_stream_list_; }
  ThreadCtxList* mut_thread_ctx_list() { return &thread_ctx_list_; }
  LivelyInstructionList* mut_lively_instruction_list() { return &lively_instruction_list_; }
  BarrierInstructionList* mut_barrier_instruction_list() { return &barrier_instruction_list_; }
  InstructionMsgMutexedList* mut_pending_msg_list() { return &pending_msg_list_; }
  InstructionMsgList* mut_local_pending_msg_list() { return &local_pending_msg_list_; }
  InstructionMsgMutexedList* mut_garbage_msg_list() { return &garbage_msg_list_; }
  StreamType2StreamRtDesc* mut_stream_type2stream_rt_desc() { return &stream_type2stream_rt_desc_; }

  // methods
  void __Init__(const VmDesc& vm_desc);
  // Returns true if old pending_instruction_list is empty
  Maybe<bool> Receive(InstructionMsgList* instr_list);
  // Returns true if old pending_instruction_list is empty
  Maybe<bool> Receive(intrusive::shared_ptr<InstructionMsg>&& instruction_msg);
  void Schedule(const ScheduleCtx& schedule_ctx);
  void Callback();
  bool SchedulerThreadUnsafeEmpty() const;
  bool SchedulerEmpty() const;
  bool CallbackEmpty() const;
  void FlushGarbageInstructions(const ScheduleCtx& schedule_ctx);
  std::string GetLivelyInstructionListDebugString(int64_t debug_cnt);

  int64_t this_machine_id() const;
  int64_t this_start_global_device_id() const {
    return this_machine_id() * vm_resource_desc().max_device_num_per_machine();
  }

  void GetCachedInstrTypeIdAndPhyInstrStream(const std::string& instr_type_name, int device_id,
                                             InstrTypeId* instr_type_id, Stream** stream);

  void GetInstrTypeIdAndSoleStream(const std::string& instr_type_name, InstrTypeId* instr_type_id,
                                   Stream** stream);

 private:
  using ReadyInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, dispatched_instruction_hook_)>;

  ReadyInstructionList* mut_ready_instruction_list() { return &ready_instruction_list_; }

  void ReleaseFinishedInstructions(const ScheduleCtx& schedule_ctx);
  void HandleLocalPending();
  void GetRewritedPendingInstructionsByWindowSize(size_t window_size,
                                                  InstructionMsgList* /*out*/ pending_instr_msgs);
  void MakeAndAppendFusedInstruction(InstructionMsgList&& fused_instr_msg_list,
                                     InstructionMsgList* /*out*/ pending_instr_msgs);
  void TryRunBarrierInstruction(const ScheduleCtx& schedule_ctx);
  void DispatchAndPrescheduleInstructions(const ScheduleCtx& schedule_ctx);
  bool OnSchedulerThread(const StreamType& stream_type);

  void ReleaseInstruction(Instruction* instruction);
  void MakeInstructions(InstructionMsg*, /*out*/ InstructionList* ret_instruction_list);

  void TryConnectInstruction(Instruction* src_instruction, Instruction* dst_instruction);
  void ConnectInstructionsByWrite(DependenceAccess* dst_access);
  void ConnectInstructionsByRead(DependenceAccess* dst_access);
  DependenceAccess* AccessMirroredObject(OperandAccessType access_type,
                                         MirroredObject* mirrored_object, Instruction* instrution);
  void ConsumeMirroredObjects(Instruction* instruction);
  void DispatchInstruction(Instruction* instruction, const ScheduleCtx& schedule_ctx);

  bool EdgeDispatchable(const Instruction* src, const Instruction* dst) const;
  bool Dispatchable(Instruction* instruction) const;
  void TryDispatchReadyInstructions();

  void LivelyInstructionListPushBack(Instruction* instruction);
  intrusive::shared_ptr<Instruction> LivelyInstructionListErase(Instruction* instruction,
                                                                const ScheduleCtx& schedule_ctx);
  void HandleLocalSchedulerProbe();
  void HandleLocalCallbackProbe();

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  VirtualMachineEngine()
      : intrusive_ref_(),
        vm_resource_desc_(),
        machine_id_range_(),
        active_stream_list_(),
        thread_ctx_list_(),
        stream_type2stream_rt_desc_(),
        pending_msg_mutex_(),
        pending_msg_list_(&pending_msg_mutex_),
        local_pending_msg_list_(),
        callback_msg_mutex_(),
        garbage_msg_list_(&callback_msg_mutex_),
        local_garbage_msg_list_(),
        ready_instruction_list_(),
        lively_instruction_list_(),
        total_inserted_instruction_cnt_(0),
        total_completed_instruction_cnt_(0),
        total_erased_instruction_cnt_(0),
        scheduler_probe_mutex_(),
        scheduler_probe_list_(&scheduler_probe_mutex_),
        local_scheduler_probe_list_(),
        callback_probe_mutex_(),
        callback_probe_list_(&callback_probe_mutex_),
        local_callback_probe_list_(),
        barrier_instruction_list_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  intrusive::shared_ptr<VmResourceDesc> vm_resource_desc_;
  Range machine_id_range_;
  // lists or maps
  // Do not change the order of the following fields
  ActiveStreamList active_stream_list_;
  ThreadCtxList thread_ctx_list_;
  StreamType2StreamRtDesc stream_type2stream_rt_desc_;
  std::mutex pending_msg_mutex_;
  InstructionMsgMutexedList pending_msg_list_;
  // local_pending_msg_list_ should be consider as the cache of pending_msg_list_.
  InstructionMsgList local_pending_msg_list_;
  std::mutex callback_msg_mutex_;
  InstructionMsgMutexedList garbage_msg_list_;
  // local_garbage_msg_list_ should be consider as the cache of garbage_msg_list_.
  InstructionMsgList local_garbage_msg_list_;
  ReadyInstructionList ready_instruction_list_;
  LivelyInstructionList lively_instruction_list_;
  size_t total_inserted_instruction_cnt_;
  size_t total_completed_instruction_cnt_;
  std::atomic<size_t> total_erased_instruction_cnt_;

  using SchedulerProbe = Probe<std::function<void(VirtualMachineEngine*)>>;
  std::mutex scheduler_probe_mutex_;
  intrusive::MutexedList<INTRUSIVE_FIELD(SchedulerProbe, probe_hook_)> scheduler_probe_list_;
  intrusive::List<INTRUSIVE_FIELD(SchedulerProbe, probe_hook_)> local_scheduler_probe_list_;
  using CallbackProbe = Probe<std::function<bool(VirtualMachineEngine*)>>;
  std::mutex callback_probe_mutex_;
  intrusive::MutexedList<INTRUSIVE_FIELD(CallbackProbe, probe_hook_)> callback_probe_list_;
  intrusive::List<INTRUSIVE_FIELD(CallbackProbe, probe_hook_)> local_callback_probe_list_;

  BarrierInstructionList barrier_instruction_list_;
  std::map<std::string, RtInstrTypeId> instr_type_name2rt_instr_type_id_;
  DependenceAccess::object_pool_type access_pool_;
  InstructionEdge::object_pool_type instruction_edge_pool_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VIRTUAL_MACHINE_ENGINE_H_
