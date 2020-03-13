#ifndef ONEFLOW_CORE_VM_VM_STREAM_RUNTIME_DESC_MSG_H_
#define ONEFLOW_CORE_VM_VM_STREAM_RUNTIME_DESC_MSG_H_

#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/vm_stream.msg.h"

namespace oneflow {

class VmStreamType;
class VmStreamDesc;

// Rt is short for Runtime
// clang-format off
OBJECT_MSG_BEGIN(VmStreamRtDesc);
  // methods
  PUBLIC void __Init__(const VmStreamDesc* vm_stream_desc);

  // fields
  OBJECT_MSG_DEFINE_PTR(const VmStreamType, vm_stream_type); 
  OBJECT_MSG_DEFINE_PTR(const VmStreamDesc, vm_stream_desc); 
  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, VmStreamTypeId, vm_stream_type_id);
  OBJECT_MSG_DEFINE_MAP_HEAD(VmStream, vm_stream_id, vm_stream_id2vm_stream);
OBJECT_MSG_END(VmStreamRtDesc);
// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_RUNTIME_DESC_MSG_H_
