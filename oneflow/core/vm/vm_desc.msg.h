#ifndef ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
#define ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_

#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/mem_zone_type_desc.msg.h"
#include "oneflow/core/vm/scheduler.msg.h"

namespace oneflow {

// clang-format off
BEGIN_OBJECT_MSG(VmDesc);
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MemZoneTypeId, mem_zone_type_id, mem_zone_type_id2desc);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VmStreamTypeId, vm_stream_type_id, remote_vm_stream_type_id2desc);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(VmStreamTypeId, vm_stream_type_id, local_vm_stream_type_id2desc);
END_OBJECT_MSG(VmDesc);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
