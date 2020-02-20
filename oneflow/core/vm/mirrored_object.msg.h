#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"
#include "oneflow/core/vm/vm_mem_zone_desc.msg.h"
#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/free_mirrored_object_handler.h"

namespace oneflow {

class VmInstructionCtx;
class MirroredObject;

// clang-format off
BEGIN_OBJECT_MSG(MirroredObjectAccess);
  // methods
  PUBLIC void __Init__(VmInstructionCtx* vm_instruction_ctx, MirroredObject* mirrored_object,
                       uint64_t logical_object_id_value, bool is_const_operand);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(bool, is_const_operand);
  OBJECT_MSG_DEFINE_RAW_PTR(VmInstructionCtx, vm_instruction_ctx);
  OBJECT_MSG_DEFINE_RAW_PTR(MirroredObject, mirrored_object);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(mirrored_object_access_link);
  OBJECT_MSG_DEFINE_LIST_LINK(vm_instr_operand_link);
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, uint64_t, logical_object_id_value);
  
END_OBJECT_MSG(MirroredObjectAccess);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(ReadOnlyAccessType);
END_FLAT_MSG(ReadOnlyAccessType);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(MirroredObjectAccessType);
  FLAT_MSG_DEFINE_ONEOF(access_type,
    FLAT_MSG_ONEOF_FIELD(ReadOnlyAccessType, read_only)
    FLAT_MSG_ONEOF_FIELD(VmStreamId, vm_stream_id_only));
END_FLAT_MSG(MirroredObjectAccessType);
// clang-format on

class LogicalObject;
// clang-format off
BEGIN_OBJECT_MSG(MirroredObject);
  // methods
  PUBLIC void __Init__(LogicalObject* logical_object, int64_t parallel_id) {
    set_logical_object(logical_object);
    set_parallel_id(parallel_id);
  }
  PUBLIC void TryResetCurrentAccessType(); 
  PUBLIC MirroredObjectAccess* GetFirstAllowedAccess();
  //fields
  OBJECT_MSG_DEFINE_FLAT_MSG(MirroredObjectAccessType, current_access_type);
  OBJECT_MSG_DEFINE_RAW_PTR(LogicalObject, logical_object);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(maybe_available_access_link);
  OBJECT_MSG_DEFINE_MAP_FLAT_MSG_KEY(int64_t, parallel_id);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, mirrored_object_access_link,
                              waiting_access_list);
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, mirrored_object_access_link,
                              holding_access_list);
  // methods
  PRIVATE bool IsFirstTwoConsumersReadOnly(); 
END_OBJECT_MSG(MirroredObject);
// clang-format on

class VmScheduler;
// clang-format off
BEGIN_OBJECT_MSG(LogicalObject);
  // methods
  PUBLIC void __Init__(const LogicalObjectId& logical_object_id,
                       VmScheduler* vm_scheduler) {
    mut_logical_object_id()->CopyFrom(logical_object_id);
    set_free_mirrored_object_handler(FreeMirroredObjectIgnoreHandler::Singleton());
    set_vm_scheduler(vm_scheduler);
  }
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const FreeMirroredObjectHandler, free_mirrored_object_handler);
  OBJECT_MSG_DEFINE_RAW_PTR(VmScheduler, vm_scheduler);
  // links
  OBJECT_MSG_DEFINE_MAP_HEAD(MirroredObject, parallel_id, parallel_id2mirrored_object);
  OBJECT_MSG_DEFINE_MAP_FLAT_MSG_KEY(LogicalObjectId, logical_object_id);
  OBJECT_MSG_DEFINE_LIST_LINK(zombie_link);
END_OBJECT_MSG(LogicalObject);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
