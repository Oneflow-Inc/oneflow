#ifndef ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_
#define ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_

#include <cstdint>
#include "oneflow/core/common/flat_msg.h"

namespace oneflow {
namespace vm {

using LogicalObjectId = int64_t;

inline int64_t GetTypeLogicalObjectId(int64_t value_logical_object_id) {
  return -value_logical_object_id;
}

inline bool IsTypeLogicalObjectId(int64_t logical_object_id) {
  return logical_object_id < 0;
}

inline bool IsValueLogicalObjectId(int64_t logical_object_id) {
  return logical_object_id > 0;
}

inline int64_t GetSelfLogicalObjectId(int64_t logical_object_id) { return logical_object_id; }

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_
