#include "oneflow/core/vm/vm_stream_runtime_desc.msg.h"

namespace oneflow {
namespace vm {

void VmStreamRtDesc::__Init__(VmStreamDesc* vm_stream_desc) {
  VmStreamTypeId vm_stream_type_id = vm_stream_desc->vm_stream_type_id();
  const VmStreamType* vm_stream_type = LookupVmStreamType(vm_stream_type_id);
  set_vm_stream_type(vm_stream_type);
  reset_vm_stream_desc(vm_stream_desc);
  set_vm_stream_type_id(vm_stream_type_id);
}

}  // namespace vm
}  // namespace oneflow
