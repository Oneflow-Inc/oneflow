#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

// clang-format off
BEGIN_FLAT_MSG(ObjectPtrValue);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, value);
END_FLAT_MSG(ObjectPtrValue);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(LogicalObjectPtrValue);
 public:
  bool operator<(const self_type& rhs) const;
  bool operator==(const self_type& rhs) const;
   
  FLAT_MSG_DEFINE_ONEOF(ptr_type,
    FLAT_MSG_ONEOF_FIELD(ObjectPtrValue, remote)
    FLAT_MSG_ONEOF_FIELD(ObjectPtrValue, local));
END_FLAT_MSG(LogicalObjectPtrValue);
// clang-format on

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

class OBJECT_MSG_TYPE(MirroredObject);

enum OBJECT_MSG_TYPE(MirroredObjectAccessType) {
  kInvalidMirroredObjectAccessType = 0,
  kMirroredObjectRead,
  kMirroredObjectWrite,
};

// clang-format off
BEGIN_OBJECT_MSG(MirroredObjectAccess);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(MirroredObjectAccessType, access_type);
  OBJECT_MSG_DEFINE_RAW_PTR(OBJECT_MSG_TYPE(MirroredObject)*, mirrored_object);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(pending_list);
  OBJECT_MSG_DEFINE_LIST_LINK(free_list);
  OBJECT_MSG_DEFINE_MAP_KEY(int32_t, vpu_operand_index);
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
  OBJECT_MSG_DEFINE_LIST_HEAD(MirroredObjectAccess, pending_list, pending_access_list);
  OBJECT_MSG_DEFINE_MAP_FLAT_MAP_KEY(LogicalObjectPtrValue, logical_ptr_val);
END_OBJECT_MSG(MirroredObject);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_H_
