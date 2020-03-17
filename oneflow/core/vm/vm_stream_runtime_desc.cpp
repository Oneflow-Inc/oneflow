#include "oneflow/core/vm/vm_stream_runtime_desc.msg.h"

namespace oneflow {
namespace vm {

void StreamRtDesc::__Init__(StreamDesc* vm_stream_desc) {
  StreamTypeId vm_stream_type_id = vm_stream_desc->vm_stream_type_id();
  const StreamType* vm_stream_type = LookupStreamType(vm_stream_type_id);
  set_vm_stream_type(vm_stream_type);
  reset_vm_stream_desc(vm_stream_desc);
  set_vm_stream_type_id(vm_stream_type_id);
}

}  // namespace vm
}  // namespace oneflow
