#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmStreamRtDesc::__Init__(const VmStreamDesc* vm_stream_desc) {
  VmStreamTypeId vm_stream_type_id = vm_stream_desc->vm_stream_type_id();
  const VmStreamType* vm_stream_type = LookupVmStreamType(vm_stream_type_id);
  set_vm_stream_type(vm_stream_type);
  set_vm_stream_desc(vm_stream_desc);
  set_vm_stream_type_id(vm_stream_type_id);
}

void VmInstruction::__Init__(VmInstructionMsg* vm_instruction_msg, VmStream* vm_stream) {
  reset_vm_instruction_msg(vm_instruction_msg);
  set_vm_stream(vm_stream);
}

void VmInstructionPackage::__Init__(VmStream* vm_stream) {
  set_vm_stream(vm_stream);
  const auto* vm_stream_type = &vm_stream->vm_thread().vm_stream_rt_desc().vm_stream_type();
  mutable_status_querier()->__Init__(
      [vm_stream_type, vm_stream](ObjectMsgAllocator* allocator, int32_t* size) {
        return vm_stream_type->NewStatusQuerier(allocator, size, vm_stream);
      });
}

}  // namespace oneflow
