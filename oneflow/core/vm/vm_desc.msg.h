#ifndef ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
#define ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_

#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/mem_zone_desc.msg.h"
#include "oneflow/core/vm/scheduler.msg.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(VmDesc);
  // methods

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(MemZoneDesc, mem_zone_type_id, mem_zone_type_id2desc);
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(StreamDesc, stream_type_id, stream_type_id2desc);
OBJECT_MSG_END(VmDesc);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
