#ifndef ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
#define ONEFLOW_CORE_VM_SCHEDULER_MSG_H_

#include <mutex>
#include "oneflow/core/vm/vpu_instruction.msg.h"

namespace oneflow {

// clang-format off
BEGIN_OBJECT_MSG(VpuSchedulerCtx);
  // methods
  void __Init__() { mutable_pending_msg_list_mutex()->__Init__(); }

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, machine_id);
  // for pending_msg_list only
  OBJECT_MSG_DEFINE_OPTIONAL(Wrapper4CppObject<std::mutex>, pending_msg_list_mutex);

  //links
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionMsg, vpu_instruction_msg_link, pending_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionMsg, vpu_instruction_msg_link, tmp_pending_msg_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link, new_vpu_instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link, waiting_vpu_instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstructionCtx, vpu_instruction_ctx_link, ready_vpu_instr_ctx_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObject, released_link, released_mirrored_object_list);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VpuCtx, vpu_id, vpu_id2vpu);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuSetCtx, vpu_set_ctx_link, vpu_set_ctx_list);
  OBJECT_MSG_DEFINE_MAP_HEAD(LogicalObject, logical_object_id, id2logical_object);
END_OBJECT_MSG(VpuSchedulerCtx);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_SCHEDULER_MSG_H_
