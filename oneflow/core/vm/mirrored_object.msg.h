#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/logical_object_id.msg.h"
#include "oneflow/core/vm/mem_zone_type_desc.msg.h"

namespace oneflow {

// clang-format off
BEGIN_OBJECT_MSG(LogicalBlobId);
END_OBJECT_MSG(LogicalBlobId);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(BlobDesc);
END_OBJECT_MSG(BlobDesc);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(Blob);
END_OBJECT_MSG(Blob);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(Operator);
END_OBJECT_MSG(Operator);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(HostMemoryBuffer);
END_OBJECT_MSG(HostMemoryBuffer);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(DeviceMemoryBuffer);
END_OBJECT_MSG(DeviceMemoryBuffer);
// clang-format on

enum OBJECT_MSG_TYPE(MirroredObjectAccessType) {
  kInvalidMirroredObjectAccessType = 0,
  kMirroredObjectRead,
  kMirroredObjectWrite,
};

class OBJECT_MSG_TYPE(VpuInstructionCtx);
class OBJECT_MSG_TYPE(MirroredObjectAccess);
class OBJECT_MSG_TYPE(MirroredObject);
// clang-format off
BEGIN_OBJECT_MSG(VpuInstrOperandAccess);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(VpuInstructionCtx)*, vpu_instruction);
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(MirroredObjectAccess)*, mirrored_object_access);
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(MirroredObject)*, mirrored_object);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(mirrored_object_access_link);
  OBJECT_MSG_DEFINE_LIST_LINK(vpu_instr_operand_link);
  OBJECT_MSG_DEFINE_SKIPLIST_FLAT_MSG_KEY(10, LogicalObjectId, logical_object_id);
END_OBJECT_MSG(VpuInstrOperandAccess);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(MirroredObjectAccess);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(MirroredObjectAccessType, access_type);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(access_pending_link);
  OBJECT_MSG_DEFINE_LIST_HEAD(VpuInstrOperandAccess, mirrored_object_access_link,
                              vpu_instr_operand_list);
END_OBJECT_MSG(MirroredObjectAccess);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(MirroredObject);
  //fields
  OBJECT_MSG_DEFINE_ONEOF(type,
    OBJECT_MSG_ONEOF_FIELD(LogicalBlobId, logical_blob_id)
    OBJECT_MSG_ONEOF_FIELD(BlobDesc, blob_desc)
    OBJECT_MSG_ONEOF_FIELD(Blob, blob)
    OBJECT_MSG_ONEOF_FIELD(Operator, op)
    OBJECT_MSG_ONEOF_FIELD(HostMemoryBuffer, host_memory_buffer)
    OBJECT_MSG_ONEOF_FIELD(DeviceMemoryBuffer, device_memory_buffer));

  // links
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, access_pending_link, access_pending_list);
  OBJECT_MSG_DEFINE_MAP_FLAT_MSG_KEY(int64_t, parallel_id);
END_OBJECT_MSG(MirroredObject);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(LogicalObject);
  // fields
  OBJECT_MSG_DEFINE_RAW_PTR(const OBJECT_MSG_TYPE(MemZoneTypeDesc)*, mem_desc);
  // links
  OBJECT_MSG_DEFINE_MAP_HEAD(MirroredObject, parallel_id, global_device_id2mirrored_object);
  OBJECT_MSG_DEFINE_MAP_FLAT_MSG_KEY(LogicalObjectId, logical_object_id);
END_OBJECT_MSG(LogicalObject);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_MSG_H_
