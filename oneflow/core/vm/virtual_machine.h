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
#ifndef ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_
#define ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_

#include <mutex>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/stream_runtime_desc.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/vm/vm_resource_desc.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/intrusive/mutexed_list.h"

namespace oneflow {

namespace vm {

struct VmDesc;
class VirtualMachine final : public intrusive::Base {
 public:
  // types
  using ActiveStreamList = intrusive::List<INTRUSIVE_FIELD(Stream, active_stream_hook_)>;
  using ThreadCtxList = intrusive::List<INTRUSIVE_FIELD(ThreadCtx, thread_ctx_hook_)>;
  using LogicalObjectDeleteList = intrusive::List<INTRUSIVE_FIELD(LogicalObject, delete_hook_)>;
  using InstructionList = intrusive::List<INTRUSIVE_FIELD(Instruction, instruction_hook_)>;
  using VmStatRunningInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, vm_stat_running_instruction_hook_)>;
  using FrontSeqInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, front_seq_compute_instr_hook_)>;
  using InstructionMsgMutextList =
      intrusive::MutexedList<INTRUSIVE_FIELD(InstructionMsg, InstructionMsg::instr_msg_hook_)>;
  using StreamTypeId2StreamRtDesc =
      intrusive::SkipList<INTRUSIVE_FIELD(StreamRtDesc, stream_type_id_)>;
  using Id2LogicalObject = intrusive::SkipList<INTRUSIVE_FIELD(LogicalObject, logical_object_id_)>;

  // Getters
  const VmResourceDesc& vm_resource_desc() const {
    if (vm_resource_desc_) { return vm_resource_desc_.Get(); }
    static const auto default_val = intrusive::make_shared<VmResourceDesc>();
    return default_val.Get();
  }
  const Range& machine_id_range() const { return machine_id_range_; }
  const std::atomic<int64_t>& flying_instruction_cnt() const { return flying_instruction_cnt_; }
  const ActiveStreamList& active_stream_list() const { return active_stream_list_; }
  const ThreadCtxList& thread_ctx_list() const { return thread_ctx_list_; }
  const LogicalObjectDeleteList& delete_logical_object_list() const {
    return delete_logical_object_list_;
  }
  const InstructionList& waiting_instruction_list() const { return waiting_instruction_list_; }
  const VmStatRunningInstructionList& vm_stat_running_instruction_list() const {
    return vm_stat_running_instruction_list_;
  }
  const FrontSeqInstructionList& front_seq_compute_instr_list() const {
    return front_seq_compute_instr_list_;
  }
  const InstructionMsgMutextList& pending_msg_list() const { return pending_msg_list_; }
  const StreamTypeId2StreamRtDesc& stream_type_id2stream_rt_desc() const {
    return stream_type_id2stream_rt_desc_;
  }
  const Id2LogicalObject& id2logical_object() const { return id2logical_object_; }
  // Setters
  VmResourceDesc* mut_vm_resource_desc() {
    if (!vm_resource_desc_) { vm_resource_desc_ = intrusive::make_shared<VmResourceDesc>(); }
    return vm_resource_desc_.Mutable();
  }
  Range* mut_machine_id_range() { return &machine_id_range_; }
  std::atomic<int64_t>* mut_flying_instruction_cnt() { return &flying_instruction_cnt_; }
  ActiveStreamList* mut_active_stream_list() { return &active_stream_list_; }
  ThreadCtxList* mut_thread_ctx_list() { return &thread_ctx_list_; }
  LogicalObjectDeleteList* mut_delete_logical_object_list() { return &delete_logical_object_list_; }
  InstructionList* mut_waiting_instruction_list() { return &waiting_instruction_list_; }
  VmStatRunningInstructionList* mut_vm_stat_running_instruction_list() {
    return &vm_stat_running_instruction_list_;
  }
  FrontSeqInstructionList* mut_front_seq_compute_instr_list() {
    return &front_seq_compute_instr_list_;
  }
  InstructionMsgMutextList* mut_pending_msg_list() { return &pending_msg_list_; }
  StreamTypeId2StreamRtDesc* mut_stream_type_id2stream_rt_desc() {
    return &stream_type_id2stream_rt_desc_;
  }
  Id2LogicalObject* mut_id2logical_object() { return &id2logical_object_; }

  // methods
  void __Init__(const VmDesc& vm_desc);
  Maybe<void> Receive(InstructionMsgList* instr_list);
  Maybe<void> Receive(intrusive::shared_ptr<InstructionMsg>&& instruction_msg);
  void Schedule();
  bool ThreadUnsafeEmpty() const;
  bool Empty() const;
  Maybe<const ParallelDesc> GetInstructionParallelDesc(const InstructionMsg&);
  MirroredObject* MutMirroredObject(int64_t logical_object_id, int64_t global_device_id);
  const MirroredObject* GetMirroredObject(int64_t logical_object_id, int64_t global_device_id);

  int64_t this_machine_id() const;
  int64_t this_start_global_device_id() const {
    return this_machine_id() * vm_resource_desc().max_device_num_per_machine();
  }

 private:
  using TmpPendingInstrMsgList = intrusive::List<INTRUSIVE_FIELD(InstructionMsg, instr_msg_hook_)>;
  using NewInstructionList = InstructionList;
  using ReadyInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, dispatched_instruction_hook_)>;

  ReadyInstructionList* mut_ready_instruction_list() { return &ready_instruction_list_; }

  bool OnSchedulerThread(const StreamType& stream_type);
  void TryRunFrontSeqInstruction();
  void ReleaseInstruction(Instruction* instruction);
  void TryReleaseFinishedInstructions(Stream* stream);
  void FilterAndRunInstructionsInAdvance(TmpPendingInstrMsgList* instr_msg_list);
  void MakeInstructions(TmpPendingInstrMsgList*, /*out*/ NewInstructionList* ret_instruction_list);
  template<int64_t (*TransformLogicalObjectId)(int64_t), typename DoEachT>
  void ForEachMirroredObject(Id2LogicalObject* id2logical_object, const Operand& operand,
                             int64_t global_device_id, const DoEachT& DoEach);
  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachConstMirroredObject(
      const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
      const ModifiedOperand<kConstModifier, mem_zone_modifier>& const_operand,
      int64_t global_device_id, const DoEachT& DoEach);
  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachConstMirroredObject(
      const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
      const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mutable_operand,
      int64_t global_device_id, const DoEachT& DoEach);
  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachMutMirroredObject(
      const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
      const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mutable_operand,
      int64_t global_device_id, const DoEachT& DoEach);
  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachMutMirroredObject(
      const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
      const ModifiedOperand<kTypeAndDataMutableModifier, mem_zone_modifier>& mut2_operand,
      int64_t global_device_id, const DoEachT& DoEach);

  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachMutMirroredObject(
      const InterpretType interpret_type, Id2LogicalObject* id2logical_object,
      const ModifiedOperand<kDeleteModifier, mem_zone_modifier>& mut2_operand,
      int64_t global_device_id, const DoEachT& DoEach);

  void ConnectInstruction(Instruction* src_instruction, Instruction* dst_instruction);
  RwMutexedObjectAccess* ConsumeMirroredObject(OperandAccessType access_type,
                                               MirroredObject* mirrored_object,
                                               Instruction* instrution);
  void ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                              NewInstructionList* new_instruction_list);
  void DispatchAndPrescheduleInstructions();
  void MoveToReadyOrWaiting(NewInstructionList* new_instruction_list);
  void DispatchInstruction(Instruction* instruction);
  void TryDeleteLogicalObjects();

  bool Dispatchable(Instruction* instruction) const;
  void TryDispatchReadyInstructions();
  void TryMoveFromWaitingToReady(Instruction* instruction);

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  VirtualMachine()
      : intrusive_ref_(),
        vm_resource_desc_(),
        machine_id_range_(),
        flying_instruction_cnt_(),
        active_stream_list_(),
        thread_ctx_list_(),
        stream_type_id2stream_rt_desc_(),
        id2logical_object_(),
        delete_logical_object_list_(),
        pending_msg_list_(),
        waiting_instruction_list_(),
        ready_instruction_list_(),
        vm_stat_running_instruction_list_(),
        front_seq_compute_instr_list_() {}
  intrusive::Ref intrusive_ref_;
  // fields
  intrusive::shared_ptr<VmResourceDesc> vm_resource_desc_;
  Range machine_id_range_;
  std::atomic<int64_t> flying_instruction_cnt_;
  // lists or maps
  // Do not change the order of the following fields
  ActiveStreamList active_stream_list_;
  ThreadCtxList thread_ctx_list_;
  StreamTypeId2StreamRtDesc stream_type_id2stream_rt_desc_;
  Id2LogicalObject id2logical_object_;
  LogicalObjectDeleteList delete_logical_object_list_;
  InstructionMsgMutextList pending_msg_list_;
  InstructionList waiting_instruction_list_;
  ReadyInstructionList ready_instruction_list_;
  VmStatRunningInstructionList vm_stat_running_instruction_list_;
  FrontSeqInstructionList front_seq_compute_instr_list_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VIRTUAL_MACHINE_H_
