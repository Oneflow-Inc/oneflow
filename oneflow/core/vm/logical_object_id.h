#ifndef ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_
#define ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_

#include <cstdint>
#include "oneflow/core/common/flat_msg.h"

namespace oneflow {
namespace vm {

using LogicalObjectId = uint64_t;

inline uint64_t GetTypeLogicalObjectId(uint64_t value_logical_object_id) {
  return static_cast<uint64_t>(-static_cast<int64_t>(value_logical_object_id));
}

inline bool IsTypeLogicalObjectId(uint64_t logical_object_id) {
  return static_cast<int64_t>(logical_object_id) < 0;
}

inline bool IsValueLogicalObjectId(uint64_t logical_object_id) {
  return static_cast<int64_t>(logical_object_id) > 0;
}

inline uint64_t GetSelfLogicalObjectId(uint64_t logical_object_id) { return logical_object_id; }

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOGICAL_OBJECT_ID_H_
