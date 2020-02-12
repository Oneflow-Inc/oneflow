#ifndef ONEFLOW_CORE_VM_MEM_DESC_MSG_H_
#define ONEFLOW_CORE_VM_MEM_DESC_MSG_H_

#include "oneflow/core/common/object_msg.h"

namespace oneflow {

using MemZoneTypeId = int32_t;

static const MemZoneTypeId kHostMemTypeId = 0;
static const MemZoneTypeId kGpuMemTypeId = 1;

// clang-format off
BEGIN_FLAT_MSG(MemZoneId);
  FLAT_MSG_DEFINE_OPTIONAL(MemZoneTypeId, mem_zone_type_id);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, parallel_id);

  // methods
  FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();
END_FLAT_MSG(MemZoneId);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(MemZoneTypeDesc);
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_machine);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, num_device);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(4, MemZoneTypeId, mem_zone_type_id);
END_OBJECT_MSG(MemZoneTypeDesc);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_DESC_MSG_H_
