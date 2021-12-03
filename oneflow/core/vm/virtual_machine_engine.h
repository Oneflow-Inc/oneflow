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
#include "oneflow/core/vm/interpret_type.h"
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

namespace oneflow {

namespace vm {

struct VmDesc;
class VirtualMachineEngine final : public intrusive::Base {
 public:
  // types
  using ActiveStreamList = intrusive::List<INTRUSIVE_FIELD(Stream, active_stream_hook_)>;
  using ThreadCtxList = intrusive::List<INTRUSIVE_FIELD(ThreadCtx, thread_ctx_hook_)>;
  using LogicalObjectDeleteList = intrusive::List<INTRUSIVE_FIELD(LogicalObject, delete_hook_)>;
  using InstructionList = intrusive::List<INTRUSIVE_FIELD(Instruction, instruction_hook_)>;
  using LivelyInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, lively_instruction_hook_)>;
  using BarrierInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, barrier_instruction_hook_)>;
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
  std::size_t flying_instruction_cnt() const {
    return pending_msg_list().thread_unsafe_size() + lively_instruction_list_.size();
  }
  const ActiveStreamList& active_stream_list() const { return active_stream_list_; }
  const ThreadCtxList& thread_ctx_list() const { return thread_ctx_list_; }
  const LogicalObjectDeleteList& delete_logical_object_list() const {
    return delete_logical_object_list_;
  }
  const LivelyInstructionList& lively_instruction_list() const { return lively_instruction_list_; }
  const BarrierInstructionList& barrier_instruction_list() const {
    return barrier_instruction_list_;
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
  ActiveStreamList* mut_active_stream_list() { return &active_stream_list_; }
  ThreadCtxList* mut_thread_ctx_list() { return &thread_ctx_list_; }
  LogicalObjectDeleteList* mut_delete_logical_object_list() { return &delete_logical_object_list_; }
  LivelyInstructionList* mut_lively_instruction_list() { return &lively_instruction_list_; }
  BarrierInstructionList* mut_barrier_instruction_list() { return &barrier_instruction_list_; }
  InstructionMsgMutextList* mut_pending_msg_list() { return &pending_msg_list_; }
  StreamTypeId2StreamRtDesc* mut_stream_type_id2stream_rt_desc() {
    return &stream_type_id2stream_rt_desc_;
  }
  Id2LogicalObject* mut_id2logical_object() { return &id2logical_object_; }

  // methods
  void __Init__(const VmDesc& vm_desc);
  // Returns true if old pending_instruction_list is empty
  Maybe<bool> Receive(InstructionMsgList* instr_list);
  // Returns true if old pending_instruction_list is empty
  Maybe<bool> Receive(intrusive::shared_ptr<InstructionMsg>&& instruction_msg);
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

  void GetCachedInstrTypeIdAndPhyInstrStream(const std::string& instr_type_name, int device_id,
                                             InstrTypeId* instr_type_id, Stream** stream);

  void GetInstrTypeIdAndSoleStream(const std::string& instr_type_name, InstrTypeId* instr_type_id,
                                   Stream** stream);

 private:
  using InstructionMsgList = intrusive::List<INTRUSIVE_FIELD(InstructionMsg, instr_msg_hook_)>;
  using ReadyInstructionList =
      intrusive::List<INTRUSIVE_FIELD(Instruction, dispatched_instruction_hook_)>;

  ReadyInstructionList* mut_ready_instruction_list() { return &ready_instruction_list_; }

  void ReleaseFinishedInstructions();
  void ReleaseFinishedRunningInstructions(Stream* stream);
  void HandlePending();
  void TryRunBarrierInstruction();
  void DispatchAndPrescheduleInstructions();
  bool OnSchedulerThread(const StreamType& stream_type);

  void ReleaseInstruction(Instruction* instruction);
  void MakeInstructions(InstructionMsg*, /*out*/ InstructionList* ret_instruction_list);
  void RunInstructionsInAdvance(InstructionMsg* instr_msg);
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

  void TryConnectInstruction(Instruction* src_instruction, Instruction* dst_instruction);
  void ConnectInstructionsByWrite(RwMutexedObjectAccess* dst_access);
  void ConnectInstructionsByRead(RwMutexedObjectAccess* dst_access);
  RwMutexedObjectAccess* AccessMirroredObject(OperandAccessType access_type,
                                              MirroredObject* mirrored_object,
                                              Instruction* instrution);
  void ConsumeMirroredObjects(Id2LogicalObject* id2logical_object, Instruction* instruction);
  void DispatchInstruction(Instruction* instruction);
  void TryDeleteLogicalObjects();

  bool EdgeDispatchable(const Instruction* src, const Instruction* dst) const;
  bool Dispatchable(Instruction* instruction) const;
  void TryDispatchReadyInstructions();

  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  VirtualMachineEngine()
      : intrusive_ref_(),
        vm_resource_desc_(),
        machine_id_range_(),
        active_stream_list_(),
        thread_ctx_list_(),
        stream_type_id2stream_rt_desc_(),
        id2logical_object_(),
        delete_logical_object_list_(),
        pending_msg_list_(),
        ready_instruction_list_(),
        lively_instruction_list_(),
        barrier_instruction_list_() {}
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
  ReadyInstructionList ready_instruction_list_;
  LivelyInstructionList lively_instruction_list_;
  BarrierInstructionList barrier_instruction_list_;
  std::map<std::string, RtInstrTypeId> instr_type_name2rt_instr_type_id_;
  RwMutexedObjectAccess::object_pool_type access_pool_;
  InstructionEdge::object_pool_type instruction_edge_pool_;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VIRTUAL_MACHINE_ENGINE_H_
