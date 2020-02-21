#ifndef ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
#define ONEFLOW_CORE_VM_SCHEDULER_MSG_H_

#include <mutex>
#include "oneflow/core/vm/vm_instruction.msg.h"

namespace oneflow {

using VmInstructionMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instruction_msg_link);
using ReadyVmInstrCtxList = OBJECT_MSG_LIST(VmInstruction, vm_instruction_link);
using MaybeAvailableAccessList = OBJECT_MSG_LIST(MirroredObject, maybe_available_access_link);
using TmpWaitingVmInstrMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instruction_msg_link);

class WaitingVmInstrCtxList;
class NewVmInstrCtxList;
class Id2LogicalObject;
class ActiveVmStreamList;
class VmDesc;

// clang-format off
BEGIN_OBJECT_MSG(VmScheduler);
  // methods
  PUBLIC void __Init__(const VmDesc& vm_desc) { __Init__(vm_desc, mut_allocator()); }
  PUBLIC void __Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator);
  PUBLIC void Receive(VmInstructionMsgList* vm_instr_list);
  PUBLIC void Schedule();

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(ObjectMsgAllocator, default_allocator);

  //links
  OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(VmInstructionMsg, vm_instruction_msg_link, waiting_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstruction, vm_instruction_link, new_vm_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstruction, vm_instruction_link, waiting_vm_instruction_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmStream, active_vm_stream_link, active_vm_stream_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmThread, vm_thread_link, vm_thread_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VmStreamRtDesc, vm_stream_type_id, vm_stream_type_id2vm_stream_rt_desc);
  OBJECT_MSG_DEFINE_MAP_HEAD(LogicalObject, logical_object_id, id2logical_object);
  OBJECT_MSG_DEFINE_LIST_HEAD(LogicalObject, zombie_link, zombie_logical_object_list);

  // methods
 private:
  using WaitingVmInstrCtxList = VmScheduler::waiting_vm_instruction_list_ObjectMsgListType;
  using NewVmInstrCtxList = VmScheduler::new_vm_instruction_list_ObjectMsgListType;
  using Id2LogicalObject = VmScheduler::id2logical_object_ObjectMsgSkipListType;
  using ActiveVmStreamList = VmScheduler::active_vm_stream_list_ObjectMsgListType;

  void ReleaseVmInstruction(VmInstruction* vm_instruction,
                            /*out*/ MaybeAvailableAccessList* maybe_available_access_list);
  void ReleaseVmInstructionPackage(VmInstructionPackage* pkg,
                                  /*out*/ MaybeAvailableAccessList* maybe_available_access_list);
  void TryReleaseFinishedVmInstructionPackages(
          VmStream* vm_stream, /*out*/ MaybeAvailableAccessList* maybe_available_access_list);
  void FilterReadyVmInstrCtx(MaybeAvailableAccessList* maybe_available_access_list,
                             WaitingVmInstrCtxList* waiting_vm_instruction_list,
                             /*out*/ ReadyVmInstrCtxList* ready_vm_instruction_list);
  void FilterAndRunControlVmInstructions(TmpWaitingVmInstrMsgList* vm_instr_msg_list);
  void MakeVmInstruction(TmpWaitingVmInstrMsgList* vm_instr_msg_list,
                         /*out*/ NewVmInstrCtxList* ret_vm_instruction_list);
  MirroredObject* FindMirroredObject(Id2LogicalObject* id2logical_object,
                                     const LogicalObjectId& logical_object_id,
                                     int64_t parallel_id);
  enum OperandAccessType { 
    kMutableOperandAccess = 0,
    kConstOperandAccess
  };

  void ConsumeMirroredObject(OperandAccessType access_type, MirroredObject* mirrored_object,
                             VmInstruction* vm_instruction);
  void ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                              NewVmInstrCtxList* new_vm_instruction_list,
                              /*out*/ MaybeAvailableAccessList* maybe_available_access_list);
  void MoveToReadyCtxListIfNoObjectOperand(NewVmInstrCtxList* new_vm_instruction_list,
                                           /*out*/ ReadyVmInstrCtxList* ready_vm_instruction_list);
  void DispatchVmInstruction(ReadyVmInstrCtxList* ready_vm_instruction_list);

END_OBJECT_MSG(VmScheduler);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
