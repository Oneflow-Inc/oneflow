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
#ifndef ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
#define ONEFLOW_CORE_VM_SCHEDULER_MSG_H_

#include <mutex>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/stream_runtime_desc.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/vm_object.msg.h"
#include "oneflow/core/vm/vm_resource_desc.msg.h"
#include "oneflow/core/common/range.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace vm {

class VmDesc;
// clang-format off
OBJECT_MSG_BEGIN(VirtualMachine);
  // methods
  OF_PUBLIC void __Init__(const VmDesc& vm_desc) { __Init__(vm_desc, mut_allocator()); }
  OF_PUBLIC void __Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator);
  OF_PUBLIC void Receive(InstructionMsgList* instr_list);
  OF_PUBLIC void Receive(ObjectMsgPtr<InstructionMsg>&& instruction_msg);
  OF_PUBLIC void Schedule();
  OF_PUBLIC bool Empty() const;
  OF_PUBLIC Maybe<ParallelDesc> GetInstructionParallelDesc(const InstructionMsg&);
  OF_PUBLIC MirroredObject* MutMirroredObject(int64_t logical_object_id, int64_t global_device_id);
  OF_PUBLIC const MirroredObject* GetMirroredObject(int64_t logical_object_id,
                                                 int64_t global_device_id);

  OF_PUBLIC int64_t this_machine_id() const;
  OF_PUBLIC int64_t this_start_global_device_id() const {
    return this_machine_id() * vm_resource_desc().max_device_num_per_machine();
  }

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VmResourceDesc, vm_resource_desc);
  OBJECT_MSG_DEFINE_STRUCT(Range, machine_id_range);
  OBJECT_MSG_DEFINE_PTR(ObjectMsgAllocator, vm_thread_only_allocator);

  //links
  OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(InstructionMsg, instr_msg_link, pending_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, waiting_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, instruction_link, ready_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, front_seq_infer_instr_link, front_seq_infer_instr_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Instruction, front_seq_compute_instr_link, front_seq_compute_instr_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Stream, active_stream_link, active_stream_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(ThreadCtx, thread_ctx_link, thread_ctx_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(StreamRtDesc, stream_type_id, stream_type_id2stream_rt_desc);
  OBJECT_MSG_DEFINE_MAP_HEAD(LogicalObject, logical_object_id, id2logical_object);

  // methods
 private:
  using TmpPendingInstrMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);
  using NewInstructionList = OBJECT_MSG_LIST(Instruction, instruction_link);
  using PrescheduledInstructionList = OBJECT_MSG_LIST(Instruction, instruction_link);
  using WaitingInstructionList = VirtualMachine::waiting_instruction_list_ObjectMsgListType;
  using ReadyInstructionList = VirtualMachine::ready_instruction_list_ObjectMsgListType;
  using Id2LogicalObject = VirtualMachine::id2logical_object_ObjectMsgSkipListType;
  using ActiveStreamList = VirtualMachine::active_stream_list_ObjectMsgListType;

  void TryRunFrontSeqInstructions();
  void ReleaseInstruction(Instruction* instruction,
                            /*out*/ ReadyInstructionList* ready_instruction_list);
  void TryReleaseFinishedInstructions(
          Stream* stream, /*out*/ ReadyInstructionList* ready_instruction_list);
  void FilterAndRunSourceInstructions(TmpPendingInstrMsgList* instr_msg_list);
  void MakeInstructions(TmpPendingInstrMsgList*, /*out*/ NewInstructionList* ret_instruction_list);
  template<int64_t (*TransformLogicalObjectId)(int64_t), typename DoEachT>
  void ForEachMirroredObject(Id2LogicalObject* id2logical_object,
                             const Operand& operand,
                             int64_t global_device_id, const DoEachT& DoEach);
  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachConstMirroredObject(const InterpretType interpret_type,
                                  Id2LogicalObject* id2logical_object,
                                  const ModifiedOperand<kConstModifier, mem_zone_modifier>& const_operand,
                                  int64_t global_device_id, const DoEachT& DoEach);
  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachConstMirroredObject(const InterpretType interpret_type,
                                  Id2LogicalObject* id2logical_object,
                                  const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mutable_operand,
                                  int64_t global_device_id, const DoEachT& DoEach);
  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachMutMirroredObject(const InterpretType interpret_type,
                                Id2LogicalObject* id2logical_object,
                                const ModifiedOperand<kDataMutableModifier, mem_zone_modifier>& mutable_operand,
                                int64_t global_device_id, const DoEachT& DoEach);
  template<OperandMemZoneModifier mem_zone_modifier, typename DoEachT>
  void ForEachMutMirroredObject(const InterpretType interpret_type,
                                Id2LogicalObject* id2logical_object,
                                const ModifiedOperand<kTypeAndDataMutableModifier, mem_zone_modifier>& mut2_operand,
                                int64_t global_device_id, const DoEachT& DoEach);
  enum OperandAccessType {
    kMutableOperandAccess = 0,
    kConstOperandAccess
  };

  void ConnectInstruction(Instruction* src_instruction, Instruction* dst_instruction);
  void ConsumeMirroredObject(OperandAccessType access_type, MirroredObject* mirrored_object,
                             Instruction* instrution);
  void ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                              NewInstructionList* new_instruction_list);
  void FilterReadyInstructions(NewInstructionList* new_instruction_list,
                         /*out*/ ReadyInstructionList* ready_instruction_list);
  void DispatchAndPrescheduleInstructions(ReadyInstructionList* ready_instruction_list);

  template<typename ReadyList, typename IsEdgeReadyT>
  void TryMoveWaitingToReady(Instruction* instruction, ReadyList* ready_list,
                             const IsEdgeReadyT& IsEdgeReady);

OBJECT_MSG_END(VirtualMachine);
// clang-format on

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
