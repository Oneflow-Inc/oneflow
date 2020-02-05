#ifndef ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_
#define ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_

#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

// clang-format off
BEGIN_FLAT_MSG(ObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, value);
END_FLAT_MSG(ObjectId);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(LogicalObjectId);
  // fields
  FLAT_MSG_DEFINE_ONEOF(ptr_type,
    FLAT_MSG_ONEOF_FIELD(ObjectId, remote)
    FLAT_MSG_ONEOF_FIELD(ObjectId, local));

  // methods
  FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();
END_FLAT_MSG(LogicalObjectId);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_
