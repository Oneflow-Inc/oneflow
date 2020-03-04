#ifndef ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_
#define ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_

#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

using LogicalObjectId = uint64_t;

// clang-format off
BEGIN_FLAT_MSG(ConstLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
END_FLAT_MSG(ConstLogicalObjectId);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(MutableLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
END_FLAT_MSG(MutableLogicalObjectId);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(ConstLocalLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
END_FLAT_MSG(ConstLocalLogicalObjectId);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(MutableLocalLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
END_FLAT_MSG(MutableLocalLogicalObjectId);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_
