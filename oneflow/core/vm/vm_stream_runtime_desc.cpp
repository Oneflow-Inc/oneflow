#include "oneflow/core/vm/vm_stream_runtime_desc.msg.h"

namespace oneflow {

void VmStreamRtDesc::__Init__(const VmStreamDesc* vm_stream_desc) {
  VmStreamTypeId vm_stream_type_id = vm_stream_desc->vm_stream_type_id();
  const VmStreamType* vm_stream_type = LookupVmStreamType(vm_stream_type_id);
  set_vm_stream_type(vm_stream_type);
  set_vm_stream_desc(vm_stream_desc);
  set_vm_stream_type_id(vm_stream_type_id);
}

}  // namespace oneflow
