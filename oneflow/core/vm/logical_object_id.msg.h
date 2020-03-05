#ifndef ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_
#define ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_

#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

using LogicalObjectId = uint64_t;

// clang-format off
FLAT_MSG_BEGIN(ConstLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
FLAT_MSG_END(ConstLogicalObjectId);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(MutableLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
FLAT_MSG_END(MutableLogicalObjectId);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(ConstLocalLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
FLAT_MSG_END(ConstLocalLogicalObjectId);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(MutableLocalLogicalObjectId);
  FLAT_MSG_DEFINE_OPTIONAL(LogicalObjectId, value);
FLAT_MSG_END(MutableLocalLogicalObjectId);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_MSG_H_
