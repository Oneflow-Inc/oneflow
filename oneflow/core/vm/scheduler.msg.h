#ifndef ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
#define ONEFLOW_CORE_VM_SCHEDULER_MSG_H_

#include <mutex>
#include "oneflow/core/vm/vm_instruction.msg.h"

namespace oneflow {

class VmDesc;

// clang-format off
BEGIN_OBJECT_MSG(VmScheduler);
  // methods
  using VmInstructionMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instr_msg_link);

  PUBLIC void __Init__(const VmDesc& vm_desc) { __Init__(vm_desc, mut_allocator()); }
  PUBLIC void __Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator);
  PUBLIC void Receive(VmInstructionMsgList* vm_instr_list);
  PUBLIC void Schedule();

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(ObjectMsgAllocator, scheduler_thread_only_allocator);

  //links
  OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(VmInstructionMsg, vm_instr_msg_link, waiting_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstrChain, vm_instr_chain_link, waiting_vm_instr_chain_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmStream, active_vm_stream_link, active_vm_stream_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmThread, vm_thread_link, vm_thread_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VmStreamRtDesc, vm_stream_type_id, vm_stream_type_id2vm_stream_rt_desc);
  OBJECT_MSG_DEFINE_MAP_HEAD(LogicalObject, logical_object_id, id2logical_object);
  OBJECT_MSG_DEFINE_LIST_HEAD(LogicalObject, zombie_link, zombie_logical_object_list);

  // methods
 private:
  using ReadyVmInstrChainList = OBJECT_MSG_LIST(VmInstrChain, vm_instr_chain_link);
  using TmpWaitingVmInstrMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instr_msg_link);
  using NewVmInstrChainList = OBJECT_MSG_LIST(VmInstrChain, vm_instr_chain_link);
  using WaitingVmInstrChainList = VmScheduler::waiting_vm_instr_chain_list_ObjectMsgListType;
  using Id2LogicalObject = VmScheduler::id2logical_object_ObjectMsgSkipListType;
  using ActiveVmStreamList = VmScheduler::active_vm_stream_list_ObjectMsgListType;

  void ReleaseVmInstruction(VmInstrChain* vm_instr_chain,
                            /*out*/ ReadyVmInstrChainList* ready_vm_instr_chain_list);
  void ReleaseVmInstrChainPackage(VmInstrChainPackage* pkg,
                                  /*out*/ ReadyVmInstrChainList* ready_vm_instr_chain_list);
  void TryReleaseFinishedVmInstrChainPackages(
          VmStream* vm_stream, /*out*/ ReadyVmInstrChainList* ready_vm_instr_chain_list);
  void FilterAndRunControlVmInstructions(TmpWaitingVmInstrMsgList* vm_instr_msg_list);
  void MakeVmInstruction(TmpWaitingVmInstrMsgList* vm_instr_msg_list,
                         /*out*/ NewVmInstrChainList* ret_vm_instr_chain_list);
  MirroredObject* FindMirroredObject(Id2LogicalObject* id2logical_object,
                                     const LogicalObjectId& logical_object_id, int64_t parallel_id);
  enum OperandAccessType { 
    kMutableOperandAccess = 0,
    kConstOperandAccess
  };

  void ConnectVmInstruction(VmInstrChain* src_vm_instr_chain, VmInstrChain* dst_vm_instr_chain);
  void ConsumeMirroredObject(OperandAccessType access_type, MirroredObject* mirrored_object,
                             VmInstruction* vm_instrution);
  void ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                              NewVmInstrChainList* new_vm_instr_chain_list,
                              /*out*/ ReadyVmInstrChainList* ready_vm_instr_chain_list);
  void DispatchVmInstruction(ReadyVmInstrChainList* ready_vm_instr_chain_list);

END_OBJECT_MSG(VmScheduler);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
