#ifndef ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_
#define ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_

#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

// clang-format off
FLAT_MSG_BEGIN(MirroredObjectId);
  // methods
  PUBLIC void __Init__(uint64_t logical_object_id_value, int64_t parallel_id) {
    set_logical_object_id_value(logical_object_id_value);
    set_parallel_id(parallel_id);
  }
  PUBLIC FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();

  // fields
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, logical_object_id_value);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, parallel_id);

FLAT_MSG_END(MirroredObjectId);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MIRRORED_OBJECT_ID_MSG_H_
