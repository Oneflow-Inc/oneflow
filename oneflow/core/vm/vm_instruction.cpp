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
  set_vm_stream_type(&vm_stream->vm_thread().vm_stream_rt_desc().vm_stream_type());
  vm_stream_type().InitVmInstructionStatus(*vm_stream, mutable_status_buffer());
}

void VmInstructionPackage::__Delete__() {
  vm_stream_type().DeleteVmInstructionStatus(vm_stream(), mut_status_buffer());
}

bool VmInstructionPackage::Done() const {
  return vm_stream_type().QueryVmInstructionStatusDone(vm_stream(), status_buffer());
}

}  // namespace oneflow
