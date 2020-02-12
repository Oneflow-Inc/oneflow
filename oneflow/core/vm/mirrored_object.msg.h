#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"
#include "oneflow/core/vm/mem_zone_type_desc.msg.h"
#include "oneflow/core/vm/vpu_type_desc.msg.h"

namespace oneflow {

// clang-format off
BEGIN_OBJECT_MSG(LogicalBlobIdObj);
END_OBJECT_MSG(LogicalBlobIdObj);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(BlobDescObj);
END_OBJECT_MSG(BlobDescObj);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(BlobObj);
END_OBJECT_MSG(BlobObj);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(OperatorObj);
END_OBJECT_MSG(OperatorObj);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(HostMemoryBuffer);
END_OBJECT_MSG(HostMemoryBuffer);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(DeviceMemoryBuffer);
END_OBJECT_MSG(DeviceMemoryBuffer);
// clang-format on

class VpuInstructionCtx;
class MirroredObject;

// clang-format off
BEGIN_OBJECT_MSG(MirroredObjectAccess);
  // methods
  PUBLIC void __Init__(VpuInstructionCtx* vpu_instruction_ctx, MirroredObject* mirrored_object,
                       uint64_t logical_object_id_value, bool is_const_operand);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(bool, is_const_operand);
  OBJECT_MSG_DEFINE_RAW_PTR(VpuInstructionCtx*, vpu_instruction_ctx);
  OBJECT_MSG_DEFINE_RAW_PTR(MirroredObject*, mirrored_object);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(mirrored_object_access_link);
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instr_operand_link);
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
    FLAT_MSG_ONEOF_FIELD(VpuId, vpu_id_only));
END_FLAT_MSG(MirroredObjectAccessType);
// clang-format on

class LogicalObject;
// clang-format off
BEGIN_OBJECT_MSG(MirroredObject);
  // methods
  PUBLIC void TryResetCurrentAccessType(); 
  PUBLIC MirroredObjectAccess* GetFirstAllowedAccess();
  //fields
  OBJECT_MSG_DEFINE_ONEOF(type,
    OBJECT_MSG_ONEOF_FIELD(LogicalBlobIdObj, logical_blob_id)
    OBJECT_MSG_ONEOF_FIELD(BlobDescObj, blob_desc)
    OBJECT_MSG_ONEOF_FIELD(BlobObj, blob)
    OBJECT_MSG_ONEOF_FIELD(OperatorObj, op)
    OBJECT_MSG_ONEOF_FIELD(HostMemoryBuffer, host_memory_buffer)
    OBJECT_MSG_ONEOF_FIELD(DeviceMemoryBuffer, device_memory_buffer));
  OBJECT_MSG_DEFINE_FLAT_MSG(MirroredObjectAccessType, current_access_type);
  OBJECT_MSG_DEFINE_RAW_PTR(LogicalObject*, logical_object);

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

// clang-format off
BEGIN_OBJECT_MSG(LogicalObject);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const MemZoneTypeDesc*, mem_desc);
  // links
  OBJECT_MSG_DEFINE_MAP_HEAD(MirroredObject, parallel_id, parallel_id2mirrored_object);
  OBJECT_MSG_DEFINE_MAP_FLAT_MSG_KEY(LogicalObjectId, logical_object_id);
END_OBJECT_MSG(LogicalObject);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
