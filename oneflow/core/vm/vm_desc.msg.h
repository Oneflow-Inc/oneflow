#ifndef ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
#define ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_

#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/scheduler.msg.h"
#include "oneflow/core/vm/vm_type.h"
#include "oneflow/core/vm/vm_resource_desc.msg.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(VmDesc);
  // methods
  PUBLIC void __Init__(const VmResourceDesc& vm_resource_desc) {
    mutable_vm_resource_desc()->CopyFrom(vm_resource_desc);
  }
  
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VmResourceDesc, vm_resource_desc);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(StreamDesc, stream_type_id, stream_type_id2desc);
OBJECT_MSG_END(VmDesc);
// clang-format on

template<VmType vm_type>
ObjectMsgPtr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
