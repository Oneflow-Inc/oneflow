#ifndef ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
#define ONEFLOW_CORE_VM_SCHEDULER_MSG_H_

#include <mutex>
#include "oneflow/core/vm/vm_instruction.msg.h"

namespace oneflow {

using VmInstructionMsgList = OBJECT_MSG_LIST(VmInstructionMsg, vm_instruction_msg_link);

// clang-format off
BEGIN_OBJECT_MSG(VpuScheduler);
  // methods
  PUBLIC void __Init__() { __Init__(mut_allocator()); }
  PUBLIC void __Init__(ObjectMsgAllocator* allocator) { set_default_allocator(allocator); }
  PUBLIC void Receive(VmInstructionMsgList* vm_instr_list);
  PUBLIC void Schedule();

  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(ObjectMsgAllocator, default_allocator);

  //links
  OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(VmInstructionMsg, vm_instruction_msg_link, waiting_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstructionMsg, vm_instruction_msg_link, tmp_waiting_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstructionCtx, vm_instruction_ctx_link, new_vm_instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstructionCtx, vm_instruction_ctx_link, waiting_vm_instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VmInstructionCtx, vm_instruction_ctx_link, ready_vm_instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObject, maybe_available_access_link, maybe_available_access_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuCtx, active_vpu_ctx_link, active_vpu_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuSetCtx, vpu_set_ctx_link, vpu_set_ctx_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VpuTypeCtx, vpu_type_id, vpu_type_id2vpu_type_ctx);
  OBJECT_MSG_DEFINE_MAP_HEAD(LogicalObject, logical_object_id, id2logical_object);
  OBJECT_MSG_DEFINE_LIST_HEAD(LogicalObject, zombie_link, zombie_logical_object_list);
END_OBJECT_MSG(VpuScheduler);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
