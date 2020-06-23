#ifndef ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
#define ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_

#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/vm_resource_desc.msg.h"
#include "oneflow/core/common/range.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(VmDesc);
  // methods
  PUBLIC void __Init__(const VmResourceDesc& vm_resource_desc) {
    __Init__(vm_resource_desc, Range(0, 1));
  }
  PUBLIC void __Init__(const VmResourceDesc& vm_resource_desc, const Range& machine_id_range) {
    mutable_vm_resource_desc()->CopyFrom(vm_resource_desc);
    *mutable_machine_id_range() = machine_id_range;
  }
  
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VmResourceDesc, vm_resource_desc);
  OBJECT_MSG_DEFINE_STRUCT(Range, machine_id_range);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(StreamDesc, stream_type_id, stream_type_id2desc);
OBJECT_MSG_END(VmDesc);
// clang-format on

ObjectMsgPtr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
