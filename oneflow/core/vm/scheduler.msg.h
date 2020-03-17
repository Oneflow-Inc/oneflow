#ifndef ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
#define ONEFLOW_CORE_VM_SCHEDULER_MSG_H_

#include <mutex>
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/stream_runtime_desc.msg.h"
#include "oneflow/core/vm/thread.msg.h"
#include "oneflow/core/vm/mirrored_object.msg.h"

namespace oneflow {
namespace vm {

class VmDesc;

// clang-format off
OBJECT_MSG_BEGIN(Scheduler);
  // methods
  using InstructionMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);

  PUBLIC void __Init__(const VmDesc& vm_desc) { __Init__(vm_desc, mut_allocator()); }
  PUBLIC void __Init__(const VmDesc& vm_desc, ObjectMsgAllocator* allocator);
  PUBLIC void Receive(InstructionMsgList* instr_list);
  PUBLIC void Receive(ObjectMsgPtr<InstructionMsg>&& instruction_msg);
  PUBLIC void Schedule();
  PUBLIC bool Empty() const;

  // fields
  OBJECT_MSG_DEFINE_PTR(ObjectMsgAllocator, scheduler_thread_only_allocator);

  //links
  OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(InstructionMsg, instr_msg_link, pending_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(InstrChain, instr_chain_link, waiting_instr_chain_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Stream, active_stream_link, active_stream_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(Thread, thread_link, thread_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(StreamRtDesc, stream_type_id, stream_type_id2stream_rt_desc);
  OBJECT_MSG_DEFINE_MAP_HEAD(LogicalObject, logical_object_id, id2logical_object);

  // methods
 private:
  using ReadyInstrChainList = OBJECT_MSG_LIST(InstrChain, instr_chain_link);
  using TmpPendingInstrMsgList = OBJECT_MSG_LIST(InstructionMsg, instr_msg_link);
  using NewInstrChainList = OBJECT_MSG_LIST(InstrChain, instr_chain_link);
  using WaitingInstrChainList = Scheduler::waiting_instr_chain_list_ObjectMsgListType;
  using Id2LogicalObject = Scheduler::id2logical_object_ObjectMsgSkipListType;
  using ActiveStreamList = Scheduler::active_stream_list_ObjectMsgListType;

  void ReleaseInstruction(InstrChain* instr_chain,
                            /*out*/ ReadyInstrChainList* ready_instr_chain_list);
  void TryReleaseFinishedInstrChains(
          Stream* stream, /*out*/ ReadyInstrChainList* ready_instr_chain_list);
  void FilterAndRunSourceControlInstructions(TmpPendingInstrMsgList* instr_msg_list);
  void MakeInstrChains(TmpPendingInstrMsgList* instr_msg_list,
                         /*out*/ NewInstrChainList* ret_instr_chain_list);
  template<typename DoEachT>
  void ForEachMirroredObject(Id2LogicalObject* id2logical_object,
                             const MirroredObjectOperand& mirrored_object_operand,
                             int64_t parallel_id, const DoEachT& DoEach);
  enum OperandAccessType {
    kMutableOperandAccess = 0,
    kConstOperandAccess
  };

  void ConnectInstruction(InstrChain* src_instr_chain, InstrChain* dst_instr_chain);
  void ConsumeMirroredObject(OperandAccessType access_type, MirroredObject* mirrored_object,
                             Instruction* instrution);
  void ConsumeMirroredObjects(Id2LogicalObject* id2logical_object,
                              NewInstrChainList* new_instr_chain_list);
  void MergeChains(NewInstrChainList* new_instr_chain_list);
  void FilterReadyChains(NewInstrChainList* new_instr_chain_list,
                         /*out*/ ReadyInstrChainList* ready_instr_chain_list);
  void DispatchInstruction(ReadyInstrChainList* ready_instr_chain_list);

OBJECT_MSG_END(Scheduler);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
