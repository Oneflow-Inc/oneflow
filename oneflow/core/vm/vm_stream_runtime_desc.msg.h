#ifndef ONEFLOW_CORE_VM_VM_STREAM_RUNTIME_DESC_MSG_H_
#define ONEFLOW_CORE_VM_VM_STREAM_RUNTIME_DESC_MSG_H_

#include "oneflow/core/vm/vm_stream_desc.msg.h"
#include "oneflow/core/vm/vm_stream.msg.h"

namespace oneflow {
namespace vm {

class StreamType;
class StreamDesc;

// Rt is short for Runtime
// clang-format off
OBJECT_MSG_BEGIN(StreamRtDesc);
  // methods
  PUBLIC void __Init__(StreamDesc* vm_stream_desc);

  // fields
  OBJECT_MSG_DEFINE_PTR(const StreamType, vm_stream_type); 
  OBJECT_MSG_DEFINE_OPTIONAL(StreamDesc, vm_stream_desc); 

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(7, StreamTypeId, vm_stream_type_id);
  OBJECT_MSG_DEFINE_MAP_HEAD(Stream, vm_stream_id, vm_stream_id2vm_stream);
OBJECT_MSG_END(StreamRtDesc);
// clang-format on

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_VM_STREAM_RUNTIME_DESC_MSG_H_
