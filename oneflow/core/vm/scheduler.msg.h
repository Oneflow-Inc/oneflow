#ifndef ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
#define ONEFLOW_CORE_VM_SCHEDULER_MSG_H_

#include <mutex>
#include "oneflow/core/vm/vpu_instruction.msg.h"

namespace oneflow {

using VpuInstructionMsgList = OBJECT_MSG_LIST(VpuInstructionMsg, vpu_instruction_msg_link);

// clang-format off
BEGIN_OBJECT_MSG(VpuScheduler);
  // methods
  PUBLIC void Receive(VpuInstructionMsgList* vpu_instr_list);
  PUBLIC void Dispatch();

  //links
  OBJECT_MSG_DEFINE_MUTEXED_LIST_HEAD(VpuInstructionMsg, vpu_instruction_msg_link, waiting_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionMsg, vpu_instruction_msg_link, tmp_waiting_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link, new_vpu_instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link, waiting_vpu_instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link, ready_vpu_instr_ctx_list);
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
